# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Forcing is a training framework for autoregressive video diffusion models that simulates the inference process during training to resolve train-test distribution mismatch. It enables real-time streaming video generation on a single RTX 4090 using KV-cached chunk-wise autoregressive rollout. Built on top of the Wan2.1 model architecture.

### Note

Don't write absolute path in markdown!!!
input absolute path from shell script, don't write in python. 

## Common Commands

### Setup
```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

### Download Models
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### Inference (CLI)
```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
Add `--i2v` for image-to-video. Distributed inference via `torchrun --nproc_per_node=N inference.py ...`.

### GUI Demo
```bash
python demo.py


```
Launches Flask server on port 5001 with WebSocket-based streaming output.
demo.py 实现了分块自回归生成（block-wise generation），通过 KV cache 在块之间传递上下文：                                                                                     
                                                                                                                                                                                
  - 默认 7 个 block × 3 帧/block = 21 帧                                                                                                                                        
  - 架构上支持增大 num_blocks 来生成更长视频                                                                                                                                    
  - local_attn_size + sink_size 机制可以限制显存使用，使显存不随视频长度无限增长                                                                                                
                                                                                                                                                                                
  如果想生成 1 分钟视频                                                                                                                                                         
                                                                                                                                                                                
  需要做两件事：                                                                                                                                                                
                                                                                                                                                                                
  1. 修改 inference.py，加入类似 demo.py 的分块循环，让它逐块生成并拼接（而不是一次性生成所有帧）                                                                               
  2. 大幅增加 num_blocks，比如 960帧 / 3帧每块 = 约 320 个 block                                                                                                                
                                                                                                                                                                                
  实际可行性考量                                                                                                                                                                
                                                                                                                                                                                
  - 显存：启用 local_attn_size 后显存是有界的，理论上可行                                                                                                                       
  - 质量：自回归生成越长，误差累积越大，画面可能逐渐退化（drift）                                                                                                               
  - 时间：320 个 block 的推理时间会非常长                                                                                                                                       
  - 这是个 1.3B 模型的蒸馏版本，本身的生成能力有限，长视频的一致性难以保证                                                                                                      
                                                                                                                                                                                
  结论                                                                                                                                                                          
                                                                                                                                                                                
  短期最可行的方式是用 demo.py 的分块机制，把 num_blocks 调大来生成更长视频。但 1 分钟（960帧）对当前模型来说是很大的挑战，质量大概率会明显下降。建议先试试生成 5-10            
  秒的视频（80-160帧），观察质量再决定是否值得跑更长的。                                 

### Training (DMD)
```bash
# Download training data first:
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts

torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
  train.py --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd --disable-wandb
```

## Architecture

### Entry Points

- **`train.py`** — Routes to trainer based on config `trainer` field: `"score_distillation"` → `ScoreDistillationTrainer`, `"diffusion"` → `DiffusionTrainer`, `"gan"` → `GANTrainer`, `"ode"` → `ODETrainer`
- **`inference.py`** — CLI batch inference. Selects `CausalInferencePipeline` (few-step, when config has `denoising_step_list`) or `CausalDiffusionInferencePipeline` (multi-step diffusion)
- **`demo.py`** — Flask GUI with WebSocket streaming, supports torch.compile, TAEHV-VAE, TensorRT VAE, FP8 quantization

### Configuration System

Two-layer OmegaConf merge: `configs/default_config.yaml` (base defaults: 21 training frames, 480×832, causal=true) is merged with task-specific configs (`self_forcing_dmd.yaml`, `self_forcing_sid.yaml`). Key config fields:

- `trainer` — selects trainer class
- `distribution_loss` — `"dmd"` or `"sid"` (selects model loss)
- `denoising_step_list` — timestep schedule (e.g., `[1000, 750, 500, 250]`)
- `num_frame_per_block` — frames per autoregressive chunk (default 3 for training)
- `warp_denoising_step` — adjusts timestep schedule using flow matching scheduler
- `image_or_video_shape` — latent shape `[B, F, C, H, W]` (e.g., `[1, 21, 16, 60, 104]`)

### Model Hierarchy (`model/`)

```
BaseModel — initializes generator, real_score, fake_score, text_encoder, VAE, scheduler
└── SelfForcingModel — adds SelfForcingTrainingPipeline for autoregressive training simulation
    ├── DMD (dmd.py) — Distribution Matching Distillation loss
    ├── SiD (sid.py) — Score Integrated Diffusion loss
    ├── CausVid (causvid.py) — CausVid baseline
    └── GAN (gan.py)
CausalDiffusion (diffusion.py) — standard diffusion loss (extends BaseModel directly)
```

`BaseModel` creates three `WanDiffusionWrapper` instances: `generator` (causal, trainable), `real_score` (bidirectional teacher, frozen), `fake_score` (bidirectional, trainable). The generator uses `CausalWanModel` with KV cache; scores use standard `WanModel`.

### Pipeline Layer (`pipeline/`)

- **`SelfForcingTrainingPipeline`** — simulates autoregressive inference during training with `inference_with_trajectory()`. Manages KV cache across chunks, gradient flow through generated context.
- **`CausalInferencePipeline`** — few-step inference with KV cache for deployment
- **`CausalDiffusionInferencePipeline`** — multi-step diffusion inference with DPM++/UniPC solver
- **`BidirectionalInferencePipeline` / `BidirectionalDiffusionInferencePipeline`** — non-causal baselines

### Wan Model Components (`wan/`)

Adapted from Wan2.1 (Apache 2.0). Key additions for Self-Forcing:

- **`wan/modules/causal_model.py`** — `CausalWanModel` extends `WanModel` with KV cache (`kv_cache1`, `kv_cache2`), block masking, local attention windows (`local_attn_size`, `sink_size`), and frame-aware RoPE frequency splits
- **`wan/modules/model.py`** — `WanModel`: DiT backbone (RoPE, RMSNorm, Flash Attention, cross-attention for text)
- **`wan/modules/vae.py`** — Wan2.1 VAE (4×8×8 stride, 16 latent channels)
- **`wan/modules/t5.py`** — UT5-XXL text encoder (512 tokens, bfloat16)
- **`wan/configs/`** — Model arch configs via `EasyDict`. `WAN_CONFIGS` dict maps model names to configs. 1.3B: 1536 dim, 12 heads, 30 layers. 14B: 5120 dim, 40 heads, 40 layers.

### Wrapper Layer (`utils/wan_wrapper.py`)

- `WanDiffusionWrapper` — wraps WanModel/CausalWanModel with scheduler, handles `is_causal` flag for model selection
- `WanTextEncoder` — wraps T5 with tokenizer
- `WanVAEWrapper` — wraps VAE with per-channel mean/std normalization

### Trainer Layer (`trainer/`)

- `ScoreDistillationTrainer` — main trainer for DMD/SiD. FSDP wrapping, distributed training, W&B logging, EMA, gradient accumulation.
- `DiffusionTrainer`, `GANTrainer`, `ODETrainer` — alternative training loops

## Key Constraints

- **Data-free training**: DMD/SiD training requires only text prompts, no video data. ODE initialization checkpoint is pre-computed.
- **Memory**: <40GB VRAM triggers `low_memory` mode with dynamic swap. `text_encoder_cpu_offload` offloads T5 to CPU.
- **Resolution**: Default 480×832. Latent shape is `[B, F, 16, 60, 104]` for this resolution.
- **Precision**: bfloat16 throughout. `mixed_precision: true` in config.
- **Flash Attention required**: All attention ops use flash_attn (v2/v3).
- **Python >=3.10, numpy==1.24.4** required.
- **Prompts**: Model works better with long, detailed prompts. Extended prompts (LLM-enriched) improve quality.
