# Self-Forcing 训练 Wan 5B 模型指南

## 概述

当前 Self-Forcing 代码基于 **Wan2.1-T2V-1.3B** 构建，本文档详细说明如何将其适配到 **Wan2.2-TI2V-5B** 模型进行训练。

核心挑战：代码中大量硬编码了 1.3B 模型的参数（维度、头数、序列长度等），需要逐一替换为 5B 的对应值。

---

## 1. 模型架构对比

| 参数 | Wan2.1-T2V-1.3B | Wan2.2-TI2V-5B | Wan2.1-T2V-14B (teacher) |
|------|-----------------|----------------|--------------------------|
| dim | 1536 | 3072 | 5120 |
| num_heads | 12 | 24 | 40 |
| head_dim | 128 | 128 | 128 |
| num_layers | 30 | 30 | 40 |
| ffn_dim | 8960 | 14336 | 13824 |
| patch_size | (1,2,2) | (1,2,2) | (1,2,2) |
| VAE | Wan2.1 (stride 4×8×8) | **Wan2.2 (stride 4×16×16)** | Wan2.1 (stride 4×8×8) |
| 默认分辨率 | 480×832 | 704×1280 | 720×1280 |
| latent 空间维度 (H×W) | 60×104 | 44×80 | 90×160 |
| frame_seq_length | 1560 (60×104/2/2) | **880** (44×80/2/2) | 3600 (90×160/2/2) |
| 21帧总 seq_len | 32760 | **18480** | 75600 |
| VAE checkpoint | Wan2.1_VAE.pth | **Wan2.2_VAE.pth** | Wan2.1_VAE.pth |

> **关键差异**: 5B 使用 Wan2.2 VAE（stride 4×16×16），空间压缩率比 Wan2.1 VAE（stride 4×8×8）大 4 倍。这导致 latent 空间尺寸、frame_seq_length、KV cache 形状等全部不同。

---

## 2. 需要准备的模型文件

### 2.1 下载 5B 模型权重

```bash
# 下载 Wan2.2-TI2V-5B 模型
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir-use-symlinks False \
    --local-dir wan_models/Wan2.2-TI2V-5B
```

目录结构应包含：
```
wan_models/Wan2.2-TI2V-5B/
├── config.json                          # diffusers 模型配置
├── diffusion_pytorch_model*.safetensors # DiT 权重
├── Wan2.2_VAE.pth                       # VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth     # T5 编码器（与 1.3B 相同）
└── google/umt5-xxl/                     # T5 tokenizer
```

### 2.2 Teacher 模型选择

DMD/SiD 训练需要一个 bidirectional teacher 模型（`real_score`）。有两个选择：

- **使用 14B 作为 teacher**: 最强的蒸馏信号，但显存需求极高
- **使用 5B bidirectional 作为 teacher**: 更节省显存，但需要 5B 的 bidirectional 权重

在 YAML 配置中通过 `real_name` 指定 teacher 模型名称。

### 2.3 ODE 初始化 checkpoint

当前 `checkpoints/ode_init.pt` 是 1.3B 的 ODE 初始化权重，**不能直接用于 5B**（参数形状不匹配）。

#### 2.3.1 ode_init.pt 的生成流程

`ode_init.pt` 通过三阶段流水线生成：

```
scripts/generate_ode_pairs.py  →  scripts/create_lmdb_iterative.py  →  train.py (trainer=ode)
     (teacher 生成 ODE 轨迹)           (打包成 LMDB 数据库)              (ODE 回归训练 causal generator)
                                                                              ↓
                                                                       checkpoints/ode_init.pt
```

**第一步：生成 ODE 轨迹对** (`scripts/generate_ode_pairs.py`)

用 bidirectional Wan2.1 模型（teacher）对每条 text prompt 做完整多步去噪：
1. 初始化纯噪声 `[1, 21, 16, 60, 104]`
2. 用 48 步 FlowMatch + CFG（guidance_scale=6.0）逐步去噪
3. 记录每一步的中间状态，得到 49 个 trajectory 点
4. 下采样到 5 个关键时间点：`[step 0, 12, 24, 36, 48]`（从纯噪声到完全干净）
5. 输出：每个 prompt 一个 `.pt` 文件，shape 为 `[1, 5, 21, 16, 60, 104]`

**第二步：打包 LMDB** (`scripts/create_lmdb_iterative.py`)

将所有 `.pt` 文件聚合成一个 LMDB 数据库，供训练高效读取。

**第三步：ODE 回归训练** (`model/ode_regression.py` + `trainer/ode.py`)

`ODERegression` 模型只含一个 causal generator（无 real_score/fake_score），训练逻辑：
1. 从 LMDB 取 ODE 轨迹 `[B, 5, 21, 16, 60, 104]`
2. 随机选一个中间时间点的 noisy latent 作为输入
3. 用 causal generator 预测 clean latent（轨迹最后一步）
4. MSE loss: `||pred - clean_latent||²`

本质是让 causal generator 学会在任意 timestep 做去噪，模仿 teacher 的 ODE 解轨迹。ODE init 为后续 DMD/SiD 蒸馏提供"warm start"。

ode_init.pt 是在对齐什么？

它已经被约束过（显式或隐式）：

causal-only

forward-time consistency

积分方向稳定


#### 2.3.2 为 5B 生成 ode_init_5B.pt

需要重跑上述三阶段流水线，所有步骤都要适配 5B：

1. **修改 `scripts/generate_ode_pairs.py`**:
   - 将 bidirectional 模型换成 5B（或继续用 14B teacher）
   - latent 形状改为 `[1, 21, 16, 44, 80]`（Wan2.2 VAE stride 4×16×16）
   - VAE 换成 Wan2.2 VAE
2. **重新跑 `scripts/create_lmdb_iterative.py`** 打包新的 LMDB
3. **用 `trainer=ode` 配置训练 5B causal generator**:
   - 需要先完成下文第 3 节中的代码修改（KV cache 形状、frame_seq_length 等）
   - 新建 `configs/ode_5B.yaml`，设置 `model_kwargs.model_name: Wan2.2-TI2V-5B`

#### 2.3.3 替代方案：跳过 ODE 初始化

如果不想跑完整的 ODE 流水线，可以直接从 5B pretrained 权重开始 DMD/SiD 训练：
- 将 `generator_ckpt` 指向 5B 的原始 checkpoint
- 修改加载逻辑使其兼容 pretrained 格式（而非 ODE trainer 保存的格式）
- 代价：训练初期收敛可能更慢，因为缺少 ODE warm-start

---

## 3. 代码修改清单

### 3.1 配置系统

#### 3.1.1 添加 5B 模型配置: `wan/configs/wan_t2v_5B.py`（新建）

```python
from easydict import EasyDict
from .shared_config import wan_shared_cfg

t2v_5B = EasyDict(__name__='Config: Wan T2V 5B')
t2v_5B.update(wan_shared_cfg)

# t5（与 1.3B 相同）
t2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae — 注意使用 Wan2.2 VAE
t2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
t2v_5B.vae_stride = (4, 16, 16)

# transformer
t2v_5B.patch_size = (1, 2, 2)
t2v_5B.dim = 3072
t2v_5B.ffn_dim = 14336
t2v_5B.freq_dim = 256
t2v_5B.num_heads = 24
t2v_5B.num_layers = 30
t2v_5B.window_size = (-1, -1)
t2v_5B.qk_norm = True
t2v_5B.cross_attn_norm = True
t2v_5B.eps = 1e-6
```

#### 3.1.2 注册配置: `wan/configs/__init__.py`

```python
from .wan_t2v_5B import t2v_5B  # 新增

WAN_CONFIGS = {
    't2v-14B': t2v_14B,
    't2v-1.3B': t2v_1_3B,
    't2v-5B': t2v_5B,        # 新增
    'i2v-14B': i2v_14B,
    't2i-14B': t2i_14B,
}
```

#### 3.1.3 训练配置: `configs/self_forcing_dmd_5B.yaml`（新建）

```yaml
generator_ckpt: checkpoints/ode_init_5B.pt
generator_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
real_name: Wan2.1-T2V-14B                    # teacher 模型，视显存选择
text_encoder_fsdp_wrap_strategy: size
denoising_step_list:
- 1000
- 750
- 500
- 250
warp_denoising_step: true
ts_schedule: false
num_train_timestep: 1000
timestep_shift: 5.0
guidance_scale: 3.0
denoising_loss_type: flow
mixed_precision: true
seed: 0
wandb_host: WANDB_HOST
wandb_key: WANDB_KEY
wandb_entity: WANDB_ENTITY
wandb_project: WANDB_PROJECT
sharding_strategy: hybrid_full
lr: 2.0e-06
lr_critic: 4.0e-07
beta1: 0.0
beta2: 0.999
beta1_critic: 0.0
beta2_critic: 0.999
data_path: prompts/vidprom_filtered_extended.txt
batch_size: 1
ema_weight: 0.99
ema_start_step: 200
total_batch_size: 64
log_iters: 50
negative_prompt: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
dfake_gen_update_ratio: 5
image_or_video_shape:            # 重要: 5B 的 latent 形状
- 1                              # batch
- 21                             # frames
- 16                             # channels
- 44                             # height (704 / 16)
- 80                             # width  (1280 / 16)
distribution_loss: dmd
trainer: score_distillation
gradient_checkpointing: true
num_frame_per_block: 3
load_raw_video: false
model_kwargs:
  model_name: Wan2.2-TI2V-5B    # 新增: 指定生成器使用 5B 模型
  timestep_shift: 5.0
```

#### 3.1.4 默认配置: `configs/default_config.yaml`

需要新增 5B 的分辨率配置，或在 5B 的 YAML 中覆盖：

```yaml
# 5B 覆盖值（在 self_forcing_dmd_5B.yaml 中添加）
height: 704
width: 1280
```

---

### 3.2 Wrapper 层修改

#### 3.2.1 `utils/wan_wrapper.py` — WanDiffusionWrapper

**问题1**: `seq_len` 硬编码为 32760 (line 141)

```python
# 当前（硬编码）
self.seq_len = 32760  # [1, 21, 16, 60, 104]

# 修改为: 从模型配置动态计算
# 5B: 21 * (44 * 80 / 2 / 2) = 21 * 880 = 18480
# 或者直接设一个足够大的值，因为 seq_len 只用于 assert
```

推荐方案 — 让 `seq_len` 作为参数传入或从模型推导：

```python
def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0
):
    super().__init__()
    # ... 模型加载代码不变 ...

    # 动态计算 seq_len
    # 从模型配置获取 dim 和 num_heads 来推断模型规模
    model_config = self.model.config if hasattr(self.model, 'config') else None
    if model_config and hasattr(model_config, 'dim'):
        # 根据模型维度推断参数
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.dim // model_config.num_heads
        self.num_layers = model_config.num_layers
    else:
        self.num_heads = 12
        self.head_dim = 128
        self.num_layers = 30

    self.seq_len = 32760  # 会在 pipeline 中被实际 seq_lens 覆盖
    self.post_init()
```

**问题2**: `adding_cls_branch()` 硬编码 1536 (line 147-154)

```python
# 当前
def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0):
    # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
    nn.Linear(atten_dim * 3 + time_embed_dim, 1536),  # ← 硬编码

# 修改为
def adding_cls_branch(self, atten_dim=None, num_class=4, time_embed_dim=0):
    if atten_dim is None:
        atten_dim = self.model.dim if hasattr(self.model, 'dim') else 1536
    self._cls_pred_branch = nn.Sequential(
        nn.LayerNorm(atten_dim * 3 + time_embed_dim),
        nn.Linear(atten_dim * 3 + time_embed_dim, atten_dim),  # ← 使用 atten_dim
        nn.SiLU(),
        nn.Linear(atten_dim, num_class)
    )
    # ...
```

#### 3.2.2 `utils/wan_wrapper.py` — WanTextEncoder

**问题**: T5 路径硬编码为 `wan_models/Wan2.1-T2V-1.3B/` (line 25, 30)

```python
# 当前
self.text_encoder.load_state_dict(
    torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", ...))
self.tokenizer = HuggingfaceTokenizer(
    name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", ...)

# 修改为: 添加 model_name 参数
class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-1.3B") -> None:
        super().__init__()
        self.text_encoder = umt5_xxl(...)
        self.text_encoder.load_state_dict(
            torch.load(f"wan_models/{model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False))
        self.tokenizer = HuggingfaceTokenizer(
            name=f"wan_models/{model_name}/google/umt5-xxl/",
            seq_len=512, clean='whitespace')
```

#### 3.2.3 `utils/wan_wrapper.py` — WanVAEWrapper

**问题**: VAE 路径和归一化常数硬编码为 Wan2.1 (line 56-71)

Wan2.2 VAE 有不同的归一化 mean/std 值和不同的模型结构。

```python
# 修改为: 支持不同 VAE
class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-1.3B"):
        super().__init__()
        if "Wan2.2" in model_name or "5B" in model_name:
            # Wan2.2 VAE 的归一化常数 (需要从模型文件或文档获取)
            mean = [...]  # TODO: 填入 Wan2.2 VAE 的 mean
            std = [...]   # TODO: 填入 Wan2.2 VAE 的 std
            from wan.modules.vae2_2 import Wan2_2_VAE  # 导入 2.2 VAE
            vae_path = f"wan_models/{model_name}/Wan2.2_VAE.pth"
            # 初始化 Wan2.2 VAE（需要查看 vae2_2.py 的工厂函数）
        else:
            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
            vae_path = f"wan_models/{model_name}/Wan2.1_VAE.pth"
            self.model = _video_vae(pretrained_path=vae_path, z_dim=16).eval().requires_grad_(False)

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
```

> **重要**: Wan2.2 VAE 的 mean/std 需要从 Wan2.2 官方代码或配置中获取。可参考 `/Wan2.2/wan/` 中的推理代码。

---

### 3.3 Pipeline 层修改 — 核心硬编码参数

以下三个 pipeline 文件都需要同样的修改模式：

- `pipeline/self_forcing_training.py`
- `pipeline/causal_inference.py`
- `pipeline/causal_diffusion_inference.py`

#### 3.3.1 `frame_seq_length` 和 `num_transformer_blocks`

**所有 pipeline 文件中都有**:
```python
self.num_transformer_blocks = 30   # 5B 也是 30，不需改
self.frame_seq_length = 1560       # 5B: 880，需要改
```

推荐方案 — 从 generator 模型动态推导：

```python
# 从 generator 获取参数
model = self.generator.model
self.num_transformer_blocks = model.num_layers       # 30 for both
# frame_seq_length = H_latent * W_latent / patch_h / patch_w
# 需要从外部传入或从 config 推算
```

由于 `frame_seq_length` 依赖于分辨率和 VAE stride，最简洁的方式是在 YAML config 中定义，或从 `image_or_video_shape` 推算：

```python
# 从 config 的 image_or_video_shape [B, F, C, H, W] 推算
H_latent = config.image_or_video_shape[3]  # 60 or 44
W_latent = config.image_or_video_shape[4]  # 104 or 80
patch_h, patch_w = 2, 2  # patch_size[1], patch_size[2]
self.frame_seq_length = (H_latent * W_latent) // (patch_h * patch_w)
# 1.3B: (60 * 104) / 4 = 1560
# 5B:   (44 * 80)  / 4 = 880
```

#### 3.3.2 KV Cache 形状

**`pipeline/self_forcing_training.py`** (line 247-248, 263-264):
```python
# 当前（硬编码 12 heads, 128 head_dim）
"k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
"v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),

# 5B 需要改为 24 heads
"k": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
"v": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
```

**推荐方案**: 参数化 num_heads 和 head_dim

```python
class SelfForcingTrainingPipeline:
    def __init__(self, ..., num_heads=12, head_dim=128):
        # ...
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, self.num_heads, self.head_dim], ...),
                "v": torch.zeros([batch_size, self.kv_cache_size, self.num_heads, self.head_dim], ...),
                # ...
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], ...),
                "v": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], ...),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
```

对 `pipeline/causal_inference.py` 和 `pipeline/causal_diffusion_inference.py` 做同样的修改。

#### 3.3.3 KV Cache 默认大小

```python
# causal_inference.py line 288
kv_cache_size = 32760  # 当前默认值 = 21 * 1560

# 5B 应改为
kv_cache_size = 18480  # 21 * 880
```

推荐动态计算：
```python
kv_cache_size = num_max_frames * self.frame_seq_length
```

---

### 3.4 CausalWanModel 中的硬编码

#### `wan/modules/causal_model.py`

**Line 76**: `CausalWanSelfAttention.max_attention_size`

```python
# 当前
self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

# 这个值在 __init__ 中计算，但 1560 是硬编码的
# 需要改为动态值，或者在运行时从实际输入推导
# 简单修复: 通过参数传入 frame_seq_length
```

**Line 508, 566, 654**: 静态方法中的 `frame_seqlen` 默认值

```python
# 当前
@staticmethod
def _prepare_blockwise_causal_attn_mask(
    device, num_frames=21, frame_seqlen=1560, ...):

# 这些是默认参数，实际调用时会传入正确值（从 x.shape 推算）
# 参见 line 890: frame_seqlen=x.shape[-2]*x.shape[-1]//(patch_size[1]*patch_size[2])
# 所以这些默认值不影响实际运行，但建议更新为通用值或移除默认值
```

---

### 3.5 inference.py 中的硬编码

**Line 86**: 图片 resize 分辨率
```python
# 当前
transforms.Resize((480, 832)),

# 5B 应改为
transforms.Resize((704, 1280)),
```

**Line 149, 162**: 噪声形状硬编码
```python
# 当前
sampled_noise = torch.randn(
    [args.num_samples, args.num_output_frames - 1, 16, 60, 104], ...)

# 5B 应改为
sampled_noise = torch.randn(
    [args.num_samples, args.num_output_frames - 1, 16, 44, 80], ...)
```

**推荐方案**: 从 config 的 `image_or_video_shape` 读取

```python
_, _, C, H, W = config.image_or_video_shape
sampled_noise = torch.randn(
    [args.num_samples, args.num_output_frames, C, H, W], ...)
```

---

### 3.6 model/base.py 中的模型加载

**Line 27-28**: 默认模型名
```python
# 当前
self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
```

这些通过 YAML 配置的 `real_name` 覆盖，所以只需在 5B 的 YAML 中正确设置即可。

**Line 39**: Text encoder 创建
```python
# 当前
self.text_encoder = WanTextEncoder()

# 修改后需要传入模型名称以定位 T5 权重路径
model_name = getattr(args, "model_kwargs", {}).get("model_name", "Wan2.1-T2V-1.3B")
self.text_encoder = WanTextEncoder(model_name=model_name)
```

**Line 42**: VAE 创建 — 同理需要传入模型名称

---

## 4. 完整修改文件清单

按优先级排列：

### P0 — 必须修改，否则无法运行

| 文件 | 位置 | 修改内容 |
|------|------|----------|
| `wan/configs/wan_t2v_5B.py` | 新建 | 5B 模型架构配置 |
| `wan/configs/__init__.py` | line 2-20 | 注册 5B config |
| `configs/self_forcing_dmd_5B.yaml` | 新建 | 5B 训练配置（含 model_kwargs.model_name） |
| `utils/wan_wrapper.py` WanDiffusionWrapper | line 141 | `seq_len` 动态化 |
| `utils/wan_wrapper.py` WanTextEncoder | line 25, 30 | T5 路径参数化 |
| `utils/wan_wrapper.py` WanVAEWrapper | line 56-71 | VAE 路径和 mean/std 适配 Wan2.2 |
| `pipeline/self_forcing_training.py` | line 29 | `frame_seq_length` 动态化 |
| `pipeline/self_forcing_training.py` | line 247-248, 263-264 | KV cache 形状: 12→24 heads |
| `pipeline/causal_inference.py` | line 34 | `frame_seq_length` 动态化 |
| `pipeline/causal_inference.py` | line 292-293, 308-309 | KV cache 形状: 12→24 heads |
| `pipeline/causal_diffusion_inference.py` | line 33 | `frame_seq_length` 动态化 |
| `wan/modules/causal_model.py` | line 76 | `max_attention_size` 中的 1560 动态化 |
| `model/base.py` | line 39, 42 | 传入 model_name 到 TextEncoder 和 VAE |

### P1 — 推理时需要修改

| 文件 | 位置 | 修改内容 |
|------|------|----------|
| `inference.py` | line 86 | I2V resize 分辨率 480×832 → 704×1280 |
| `inference.py` | line 149, 162 | 噪声形状 [16,60,104] → [16,44,80] |
| `demo.py` | 多处 | latent 形状硬编码 |
| `long.py` | 多处 | frame_seq_length 引用 |

### P2 — GAN 分支（如果使用）

| 文件 | 位置 | 修改内容 |
|------|------|----------|
| `utils/wan_wrapper.py` adding_cls_branch | line 147-154 | `atten_dim` 1536→3072 |

---

## 5. 推荐的重构方案

与其逐一修改硬编码值，建议做一次系统性重构，让所有模型相关参数从 config 自动推导：

### 5.1 在 config 中添加模型元数据

在 YAML config 中新增：
```yaml
# 模型架构参数（从 WAN_CONFIGS 自动填充或手动指定）
model_dim: 3072
model_num_heads: 24
model_head_dim: 128
model_num_layers: 30
vae_type: "wan2.2"   # "wan2.1" or "wan2.2"
```

### 5.2 在 pipeline 初始化时统一推导

```python
# 在所有 pipeline 的 __init__ 中:
latent_shape = args.image_or_video_shape  # [B, F, C, H, W]
self.frame_seq_length = (latent_shape[3] * latent_shape[4]) // (2 * 2)  # patch_size
self.num_transformer_blocks = args.model_num_layers  # 或从 generator.model 获取
self.num_heads = args.model_num_heads
self.head_dim = args.model_head_dim
```

---

## 6. 显存估算

### 1.3B 训练显存（参考）

- Generator (causal, trainable): ~5.2 GB
- Real score (bidirectional, frozen): ~5.2 GB (若用 14B teacher: ~56 GB)
- Fake score (bidirectional, trainable): ~5.2 GB
- T5 text encoder: ~18 GB
- VAE: ~0.5 GB
- KV Cache: ~2-4 GB
- 激活值 + 梯度: ~10-20 GB
- **总计**: ~40-50 GB（1.3B teacher）/ ~90+ GB（14B teacher）

### 5B 训练显存估算

- Generator (causal, trainable): ~20 GB
- Real score (若用 5B bidirectional): ~20 GB
- Fake score (trainable): ~20 GB
- T5 text encoder: ~18 GB
- VAE: ~0.5 GB
- KV Cache: ~1-2 GB（frame_seq_length 更小，但 num_heads 更多）
- 激活值 + 梯度: ~30-50 GB
- **总计**: ~110-130 GB

### 显存优化建议

1. **FSDP 分片**: 已有支持（`sharding_strategy: hybrid_full`），多卡分片模型参数
2. **text_encoder_cpu_offload**: 将 T5 放 CPU，省 ~18 GB
3. **gradient_checkpointing**: 已启用，节省激活值显存
4. **减小训练分辨率**: 可以先用 480×832 训练 5B（需重算 latent 形状: Wan2.2 VAE stride=16→30×52→frame_seq_length=390）
5. **最低配置**: 8×A100 80GB 或 8×H100 80GB（使用 FSDP）

---

## 7. 训练启动命令

```bash
# 8 节点 × 8 GPU
torchrun --nnodes=8 --nproc_per_node=8 \
    --rdzv_id=5235 --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/self_forcing_dmd_5B.yaml \
    --logdir logs/self_forcing_dmd_5B \
    --disable-wandb

# 单节点 8 GPU（显存受限，需要更激进的 FSDP）
torchrun --nproc_per_node=8 \
    train.py \
    --config_path configs/self_forcing_dmd_5B.yaml \
    --logdir logs/self_forcing_dmd_5B \
    --disable-wandb
```

---

## 8. 验证检查项

完成代码修改后，逐步验证：

1. **Config 加载**: 确认 `WAN_CONFIGS['t2v-5B']` 返回正确参数
2. **模型加载**: `CausalWanModel.from_pretrained("wan_models/Wan2.2-TI2V-5B/")` 成功
3. **KV Cache 形状**: 确认 cache tensor 为 `[B, kv_size, 24, 128]`
4. **frame_seq_length**: 确认为 880（704×1280 / 16 / 16 / 2 / 2）
5. **VAE 编解码**: 输入 `[1, 3, 21, 704, 1280]` → latent `[1, 21, 16, 44, 80]` → 重建 `[1, 3, 21, 704, 1280]`
6. **单步前向传播**: 跑一次 generator forward，确认输出形状正确
7. **单步训练**: 确认梯度正常回传

```python
# 快速验证脚本
import torch
from wan.modules.causal_model import CausalWanModel

model = CausalWanModel.from_pretrained("wan_models/Wan2.2-TI2V-5B/")
print(f"dim={model.dim}, heads={model.num_heads}, layers={model.num_layers}")
# 应输出: dim=3072, heads=24, layers=30
```

---

## 9. 已知风险与注意事项

1. **Wan2.2 VAE 兼容性**: Self-Forcing 的 `wan/modules/vae.py` 只包含 Wan2.1 VAE 实现。需要将 Wan2.2 VAE（`/Wan2.2/wan/modules/vae2_2.py`）复制到 Self-Forcing 的 `wan/modules/` 目录中。

2. **flex_attention 编译**: `causal_model.py` line 23-24 写了 `mode="max-autotune-no-cudagraphs"` 并注释说这是 1.3B 的特殊需求。5B 模型的 head_dim=128 是标准的，可能需要调整编译模式。

3. **Teacher 模型 VAE 不匹配**: 如果 teacher 用 14B（Wan2.1 VAE），generator 用 5B（Wan2.2 VAE），两者的 latent 空间不同，**DMD loss 计算可能有问题**。需要确保 teacher 和 generator 在相同的 latent 空间中操作，或者使用同样使用 Wan2.2 VAE 的 teacher。

4. **ODE 初始化**: 1.3B 的 `ode_init.pt` 权重形状与 5B 不兼容，不能直接加载。需要重新生成或跳过。

---

## 10. `wan/modules/model.py` 详细修改指南

### 10.1 背景：TI2V vs I2V 的区别

Wan2.2-TI2V-5B 使用 `model_type='ti2v'`，它与 I2V-14B 的实现方式**完全不同**：

| 特性 | I2V-14B (`model_type='i2v'`) | TI2V-5B (`model_type='ti2v'`) |
|------|------------------------------|-------------------------------|
| 图片嵌入方式 | CLIP 特征 (`clip_fea`) + 专用 `WanI2VCrossAttention` | VAE 编码首帧作为 inpainting 条件 (`y`) |
| 交叉注意力 | `WanI2VCrossAttention` (有额外的 k_img, v_img 投影) | `WanT2VCrossAttention` (与 T2V 相同) |
| 额外模块 | `self.img_emb = MLPProj(1280, dim)` | 无 |
| 首帧条件 | `clip_fea` (CLIP 特征) | `y` (VAE 编码的首帧 latent) |

**核心结论**: TI2V 在模型结构上与 T2V 几乎相同，只是在推理时通过 `y` 参数传入首帧条件。

### 10.2 当前 Self-Forcing model.py 的限制

当前 `Self-Forcing/wan/modules/model.py` (line 563) 只支持 `t2v` 和 `i2v`：

```python
assert model_type in ['t2v', 'i2v']
```

而 Wan2.2-TI2V-5B 的 `config.json` 指定：
```json
{
  "model_type": "ti2v",
  ...
}
```

### 10.3 具体代码修改

#### 修改 1: 扩展支持的 model_type (Line 563)

```python
# 当前
assert model_type in ['t2v', 'i2v']

# 修改为
assert model_type in ['t2v', 'i2v', 'ti2v']
```

#### 修改 2: 添加 ti2v 交叉注意力类型映射 (Line 269-272)

```python
# 当前
WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

# 修改为（添加 ti2v，它使用与 t2v 相同的交叉注意力）
WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'ti2v_cross_attn': WanT2VCrossAttention,  # 新增: TI2V 使用 T2V 的交叉注意力
}
```

#### 修改 3: 更新交叉注意力类型选择逻辑 (Line 594-595)

```python
# 当前
cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

# 修改为
if model_type == 't2v' or model_type == 'ti2v':
    cross_attn_type = 't2v_cross_attn'
else:
    cross_attn_type = 'i2v_cross_attn'
```

或者更简洁的写法：
```python
cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
```

#### 修改 4: 更新 img_emb 初始化逻辑 (Line 615-616)

```python
# 当前
if model_type == 'i2v':
    self.img_emb = MLPProj(1280, dim)

# 保持不变 — TI2V 不需要 img_emb
# 因为 TI2V 不使用 CLIP 特征，而是通过 y 参数传入 VAE 编码的首帧
```

#### 修改 5: 更新 forward 中的断言逻辑 (Line 672-673)

```python
# 当前
if self.model_type == 'i2v':
    assert clip_fea is not None and y is not None

# 修改为: TI2V 只需要 y，不需要 clip_fea
if self.model_type == 'i2v':
    assert clip_fea is not None and y is not None
elif self.model_type == 'ti2v':
    # TI2V 的 y 是可选的：
    # - T2V 模式: y=None（纯文本生成）
    # - I2V 模式: y!=None（首帧条件生成）
    pass  # 无断言，y 可以是 None 或非 None
```

### 10.4 完整的修改 diff

```diff
--- a/wan/modules/model.py
+++ b/wan/modules/model.py
@@ -266,6 +266,7 @@ class WanI2VCrossAttention(WanSelfAttention):
 WAN_CROSSATTENTION_CLASSES = {
     't2v_cross_attn': WanT2VCrossAttention,
     'i2v_cross_attn': WanI2VCrossAttention,
+    'ti2v_cross_attn': WanT2VCrossAttention,  # TI2V uses same cross-attn as T2V
 }
 
 
@@ -560,7 +561,7 @@ class WanModel(ModelMixin, ConfigMixin):
 
         super().__init__()
 
-        assert model_type in ['t2v', 'i2v']
+        assert model_type in ['t2v', 'i2v', 'ti2v']
         self.model_type = model_type
 
         self.patch_size = patch_size
@@ -591,7 +592,10 @@ class WanModel(ModelMixin, ConfigMixin):
         self.time_projection = nn.Sequential(
             nn.SiLU(), nn.Linear(dim, dim * 6))
 
-        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
+        # TI2V uses same cross-attention as T2V (no CLIP feature projection)
+        # I2V uses special cross-attention with separate image K/V projections
+        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
+
         self.blocks = nn.ModuleList([
             WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                               window_size, qk_norm, cross_attn_norm, eps)
@@ -612,6 +616,7 @@ class WanModel(ModelMixin, ConfigMixin):
         ],
             dim=1)
 
+        # Only I2V needs the CLIP image embedding projection (TI2V does not)
         if model_type == 'i2v':
             self.img_emb = MLPProj(1280, dim)
 
@@ -669,6 +674,8 @@ class WanModel(ModelMixin, ConfigMixin):
         """
         if self.model_type == 'i2v':
             assert clip_fea is not None and y is not None
+        # TI2V: y is optional (None for T2V mode, not-None for I2V mode)
+        # No assertion needed for ti2v
         # params
         device = self.patch_embedding.weight.device
         if self.freqs.device != device:
```

### 10.5 同步修改 causal_model.py

`wan/modules/causal_model.py` 也需要类似修改。关键位置：

#### Line 562-563 (CausalWanModel.__init__)
```python
# 当前
assert model_type in ['t2v', 'i2v']

# 修改为
assert model_type in ['t2v', 'i2v', 'ti2v']
```

#### Line 595-596 (cross_attn_type 选择)
```python
# 当前
cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

# 修改为
cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
```

#### Line 615-616 (img_emb 初始化)
```python
# 保持不变 — TI2V 不需要 img_emb
if model_type == 'i2v':
    self.img_emb = MLPProj(1280, dim)
```

### 10.6 TI2V 使用方式

TI2V 模型支持两种生成模式：

**1. 纯文本生成 (T2V 模式)**
```python
# 不传入 y 参数，等同于 T2V
output = model(x=noise, t=timesteps, context=text_embeddings, seq_len=seq_len)
```

**2. 首帧条件生成 (I2V 模式)**
```python
# 传入 y 参数（VAE 编码的首帧 latent）
# 注意: 不需要 clip_fea，这与 I2V-14B 不同
output = model(x=noise, t=timesteps, context=text_embeddings, seq_len=seq_len, y=first_frame_latent)
```

在训练代码中，`y` 参数的处理逻辑（与当前 I2V 相同）：
```python
# model.py line 679-680
if y is not None:
    x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
```

### 10.7 验证修改正确性

```python
# 验证脚本
import torch
from wan.modules.model import WanModel

# 加载 TI2V 模型
model = WanModel.from_pretrained("wan_models/Wan2.2-TI2V-5B/")

# 检查模型类型和配置
print(f"model_type: {model.model_type}")  # 应输出: ti2v
print(f"dim: {model.dim}")                # 应输出: 3072
print(f"num_heads: {model.num_heads}")    # 应输出: 24
print(f"num_layers: {model.num_layers}")  # 应输出: 30

# 检查没有 img_emb (因为 TI2V 不需要)
print(f"has img_emb: {hasattr(model, 'img_emb')}")  # 应输出: False

# 检查交叉注意力类型（应该是 T2V 类型）
block = model.blocks[0]
print(f"cross_attn type: {type(block.cross_attn).__name__}")  # 应输出: WanT2VCrossAttention
```

---

## 11. 5B 模型分片加载问题（重点）

### 11.1 问题现象

5B 模型约 20GB，被分成 3 个 safetensors 文件：
```
diffusion_pytorch_model-00001-of-00003.safetensors
diffusion_pytorch_model-00002-of-00003.safetensors
diffusion_pytorch_model-00003-of-00003.safetensors
diffusion_pytorch_model.safetensors.index.json
```

使用 `WanModel.from_pretrained()` 加载时遇到各种错误。

### 11.2 尝试的方案及失败原因

#### 方案 1：直接 `from_pretrained` + `.to(device)`

```python
model = WanModel.from_pretrained(model_path)
model.to(device)
```

**错误**: `NotImplementedError: Cannot copy out of meta tensor; no data!`

**原因**: diffusers 使用 accelerate 的 meta tensor 机制。模型先在 meta device（虚拟设备，无实际数据）上初始化，然后 `load_checkpoint_and_dispatch` 按需加载权重。但在调用 `.to(device)` 时，meta tensor 没有数据无法迁移。

#### 方案 2：使用 `device_map` 参数

```python
model = WanModel.from_pretrained(model_path, device_map=device)
```

**错误**: `ValueError: weight is on the meta device, we need a value to put in on {device}`

**原因**: 在分布式环境下（torchrun 启动 4 个进程），每个进程有不同的 device ID (0,1,2,3)。accelerate 的 device_map 机制不支持这种多进程同时加载的场景。每个进程都尝试初始化 meta tensor 然后 dispatch 到自己的 GPU，但 dispatch 逻辑没有正确处理。

#### 方案 3：禁用 `low_cpu_mem_usage`

```python
model = WanModel.from_pretrained(model_path, low_cpu_mem_usage=False)
```

**错误**: `TypeError: expected str, bytes or os.PathLike object, not NoneType`

**原因**: 当 `low_cpu_mem_usage=False` 时，diffusers 尝试找单个 checkpoint 文件加载。但 5B 是分片存储的，它找不到 `diffusion_pytorch_model.safetensors`（不存在），返回 None，导致后续文件打开失败。

### 11.3 根本原因

1. **模型太大**: 5B 约 20GB，单 GPU 显存不够装完整模型 + 梯度
2. **分片存储格式**: safetensors 分片格式需要特殊处理
3. **diffusers 版本限制**: Self-Forcing 使用的 diffusers 版本对分片模型支持不完善
4. **分布式环境冲突**: torchrun 多进程与 accelerate 的自动设备映射机制冲突

### 11.4 可行的解决方案

#### 方案 A：手动加载分片模型（推荐）

绕过 diffusers 的 `from_pretrained`，直接用 safetensors 库加载：

```python
from safetensors.torch import load_file
import json
import os

def load_sharded_safetensors(model_path):
    """手动加载分片 safetensors 模型"""
    index_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")

    with open(index_path, 'r') as f:
        index = json.load(f)

    # 获取所有分片文件
    shard_files = sorted(set(index["weight_map"].values()))

    # 逐个加载合并
    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        print(f"Loading {shard_file}...")
        shard_dict = load_file(shard_path)
        state_dict.update(shard_dict)

    return state_dict

# 使用方式
from wan.modules.model import WanModel
import json

# 1. 读取 config
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)

# 2. 创建空模型
model = WanModel(**config)

# 3. 加载权重
state_dict = load_sharded_safetensors(model_path)
model.load_state_dict(state_dict)

# 4. 移动到 GPU
model.to(device)
```

#### 方案 B：单 GPU 模式

放弃多 GPU 并行，改用单 GPU：

```bash
# gen_ode_5B.sh
export CUDA_VISIBLE_DEVICES=0
python scripts/generate_ode_pairs5B.py \
    --output_folder ode_pairs_5B/ \
    --caption_path prompts/vidprom_filtered_extended.txt \
    --model_path /path/to/Wan2.2-TI2V-5B
```

然后在代码中用单进程加载：
```python
model = WanModel.from_pretrained(model_path, device_map="cuda:0")
```

缺点：生成速度慢，需要更多时间。

#### 方案 C：参考原版 Wan2.2 加载方式

原版 `wan/textimage2video.py` 的流程：
1. `init_on_cpu=True` — 在 CPU 上初始化
2. `from_pretrained` 加载
3. 推理时再 `.to(device)`

关键：原版不在多进程环境下每个进程都加载完整模型，而是使用 FSDP 分片或单进程。

#### 方案 D：升级 diffusers 版本

新版 diffusers (>=0.28) 对分片模型支持更好。尝试：
```bash
pip install diffusers>=0.28
```

但可能引入其他兼容性问题。

### 11.5 推荐实现

在 `scripts/generate_ode_pairs5B.py` 中使用方案 A：

```python
class WanDiffusionWrapper5B(torch.nn.Module):
    def __init__(self, model_path, timestep_shift=5.0):
        super().__init__()

        # 手动加载分片模型
        self.model = self._load_sharded_model(model_path)
        self.model.eval()
        # ... 其他初始化代码 ...

    def _load_sharded_model(self, model_path):
        from safetensors.torch import load_file
        import json

        # 读取 config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # 创建模型
        model = WanModel(**config)

        # 加载分片权重
        index_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        state_dict = {}
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(model_path, shard_file)
            state_dict.update(load_file(shard_path))

        model.load_state_dict(state_dict)
        return model
```

### 11.6 架构差异总结

| 特性 | Wan2.1 1.3B | Wan2.2 TI2V-5B |
|------|-------------|----------------|
| 模型大小 | ~2.6GB | ~20GB |
| 存储格式 | 单个 .pth 或 .safetensors | 分片 safetensors (3 个文件) |
| model_type | t2v | ti2v |
| latent channels | 16 | 48 |
| VAE stride | (4, 8, 8) | (4, 16, 16) |
| 分辨率 | 480×832 | 704×1280 |
| 加载复杂度 | 简单 | 需要手动处理分片 |

### 11.7 img_emb 参数不匹配问题

#### 问题现象

使用 `device_map="auto"` 加载时报错：
```
ValueError: weight is on the meta device, we need a `value` to put in on 0.
```

#### 原因分析

1. **model.py 的修改引入了问题**：
   我们之前修改了 `wan/modules/model.py` 第 615-616 行：
   ```python
   # 修改前
   if model_type == 'i2v':
       self.img_emb = MLPProj(1280, dim)

   # 修改后（错误）
   if model_type in ['i2v', 'ti2v']:
       self.img_emb = MLPProj(1280, dim)
   ```

2. **TI2V 不需要 img_emb**：
   - I2V-14B 使用 CLIP 特征作为图片条件，需要 `img_emb` 投影层
   - TI2V-5B 使用 VAE 编码的首帧作为条件（通过 `y` 参数），**不需要** `img_emb`

3. **checkpoint 中没有 img_emb 权重**：
   - 当模型初始化时，`img_emb` 在 meta device 上创建
   - accelerate 尝试从 checkpoint 加载权重，但找不到 `img_emb.*` 的权重
   - 导致 `img_emb` 的参数仍在 meta device 上，无法 dispatch 到真实设备

4. **accelerate 的 meta tensor 机制**：
   ```
   模型初始化 (meta device) → 加载 checkpoint → dispatch 到真实设备
                                    ↓
                           img_emb 没有对应权重
                                    ↓
                           img_emb 仍在 meta device
                                    ↓
                           dispatch 失败
   ```

#### 解决方案

**方案 1：修复 model.py 的 img_emb 条件判断（推荐）**

```python
# wan/modules/model.py line 615-616
# TI2V 不需要 img_emb，只有 I2V 需要
if model_type == 'i2v':  # 不要包含 'ti2v'
    self.img_emb = MLPProj(1280, dim)
```

**方案 2：手动加载避免 meta tensor**

使用 safetensors 手动加载，完全绕过 accelerate 的 meta tensor 机制：

```python
from safetensors.torch import load_file
import json

def load_5b_model(model_path, device):
    # 1. 读取 config
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    # 2. 在真实设备上创建模型（不用 meta tensor）
    model = WanModel(**config)

    # 3. 加载分片权重
    index_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    state_dict = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        shard_dict = load_file(os.path.join(model_path, shard_file))
        state_dict.update(shard_dict)

    # 4. 加载权重（strict=False 允许多余参数）
    model.load_state_dict(state_dict, strict=False)

    # 5. 移动到目标设备
    model.to(device)
    return model
```

#### TI2V vs I2V 的关键区别

| 特性 | I2V-14B | TI2V-5B |
|------|---------|---------|
| 图片条件方式 | CLIP 特征 (`clip_fea`) | VAE 编码首帧 (`y`) |
| 需要 img_emb | **是** | **否** |
| 交叉注意力 | `WanI2VCrossAttention` | `WanT2VCrossAttention` |
| checkpoint 包含 img_emb | **是** | **否** |

TI2V 本质上是 T2V + 首帧 inpainting 条件，模型结构与 T2V 相同。
