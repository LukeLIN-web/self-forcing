"""
Generate ODE trajectory pairs for Self-Forcing training using Wan2.2 TI2V-5B model.

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 --master_port=29501 scripts/generate_ode_pairs5B.py \
        --output_folder ode_pairs_5B/ \
        --caption_path prompts/vidprom_filtered_extended.txt

Output format:
    Each .pt file contains {prompt: tensor} where tensor has shape:
    [num_snapshots, 1, num_frames, 48, height, width]
"""

from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.dataset import TextDataset
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel
from wan.modules.t5 import umt5_xxl
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


# 5B model path (set via --model_path argument)
MODEL_PATH = None

# 5B model latent config (704x1280 resolution)
# VAE stride: (4, 16, 16) -> latent: frames/4, height/16, width/16
NUM_FRAMES = 21  # Use 21 frames like 1.3B for compatibility, or change to 31 for 5B native
LATENT_CHANNELS = 48  # 5B uses 48 channels
LATENT_HEIGHT = 44   # 704 / 16 = 44
LATENT_WIDTH = 80    # 1280 / 16 = 80

# Negative prompt (Chinese aesthetic guidelines)
NEGATIVE_PROMPT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'


class WanTextEncoder5B(torch.nn.Module):
    """Text encoder for 5B model."""

    def __init__(self, model_path=MODEL_PATH):
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)

        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                map_location='cpu',
                weights_only=False
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_path, "google/umt5-xxl/"),
            seq_len=512,
            clean='whitespace'
        )

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts):
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        return {"prompt_embeds": context}


class WanDiffusionWrapper5B(torch.nn.Module):
    """Diffusion wrapper for 5B model."""

    def __init__(self, model_path=MODEL_PATH, timestep_shift=5.0, device=None):
        super().__init__()

        # Load sharded model - use "auto" device_map, then move to specific device
        self.model = WanModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float32
        )
        self.model.eval()

        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # seq_len for 5B: num_frames * (height/patch) * (width/patch)
        # patch_size = (1, 2, 2), so: 21 * 22 * 40 = 18480
        self.seq_len = NUM_FRAMES * (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred, xt, timestep):
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(self, noisy_image_or_video, conditional_dict, timestep):
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Model expects [B, C, F, H, W], input is [B, F, C, H, W]
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0


def init_model(device, model_path):
    print(f"Loading 5B model from {model_path}...")

    # Pass device to wrapper for device_map loading
    model = WanDiffusionWrapper5B(model_path=model_path, device=device)
    encoder = WanTextEncoder5B(model_path=model_path).to(device).to(torch.float32)
    model.model.requires_grad_(False)

    scheduler = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True
    )
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    unconditional_dict = encoder(text_prompts=[NEGATIVE_PROMPT])

    print(f"Model loaded. Latent shape: [1, {NUM_FRAMES}, {LATENT_CHANNELS}, {LATENT_HEIGHT}, {LATENT_WIDTH}]")

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--caption_path", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=5.0)  # 5B uses 5.0
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--model_path", type=str, required=True, help="Path to Wan2.2-TI2V-5B model")

    args = parser.parse_args()

    # Support both single GPU and distributed modes
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        launch_distributed_job()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(device=device, model_path=args.model_path)

    dataset = TextDataset(args.caption_path)

    os.makedirs(args.output_folder, exist_ok=True)

    if rank == 0:
        print(f"Total prompts: {len(dataset)}")
        print(f"Output folder: {args.output_folder}")

    for index in tqdm(
        range(int(math.ceil(len(dataset) / world_size))),
        disable=rank != 0
    ):
        prompt_index = index * world_size + rank
        if prompt_index >= len(dataset):
            continue

        batch = dataset[prompt_index]
        prompt = batch["prompts"] if isinstance(batch, dict) else batch

        conditional_dict = encoder(text_prompts=[prompt])

        # 5B latent shape: [B, F, C, H, W]
        latents = torch.randn(
            [1, args.num_frames, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH],
            dtype=torch.float32,
            device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(scheduler.timesteps):
            timestep = t * torch.ones(
                [1, args.num_frames], device=device, dtype=torch.float32
            )

            noisy_input.append(latents)

            _, x0_pred_cond = model(latents, conditional_dict, timestep)
            _, x0_pred_uncond = model(latents, unconditional_dict, timestep)

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, args.num_frames], device=device, dtype=torch.long
                ).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        # Stack all snapshots and select key steps
        noisy_inputs = torch.stack(noisy_input, dim=1)
        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]

        torch.save(
            {prompt: noisy_inputs.cpu().detach()},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    if is_distributed:
        dist.barrier()

    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
