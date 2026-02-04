# Self-Forcing DMD/SiD 训练架构详解

## 1. 总览

Self-Forcing 采用 **score distillation（分数蒸馏）** 的方式训练自回归视频扩散模型。核心思想是：在训练时模拟推理的自回归过程，通过 KV cache 分块生成视频，并利用 teacher 模型的分布信息指导 generator 学习。

整个训练是 **data-free** 的——只需要文本 prompt，不需要视频数据集。

### 三个模型

| 模型 | 类型 | 是否可训练 | 作用 |
|------|------|-----------|------|
| **generator** | Causal (CausalWanModel + KV cache) | 可训练 | 自回归生成视频帧 |
| **real_score** | Bidirectional (WanModel) | 冻结 | 教师模型，提供真实分布信号 |
| **fake_score** | Bidirectional (WanModel) | 可训练 | 评论家 (critic)，学习预测 generator 生成样本的去噪结果 |

---

## 2. 入口与配置

### train.py

```
train.py
  → 加载 configs/default_config.yaml（基础默认值）
  → 与任务配置合并（如 configs/self_forcing_dmd.yaml）
  → 根据 config.trainer 选择 Trainer：
      "score_distillation" → ScoreDistillationTrainer（DMD/SiD 都走这里）
  → trainer.train() 开始训练
```

`train.py:39` 中，`config.trainer == "score_distillation"` 时创建 `ScoreDistillationTrainer`。

### ScoreDistillationTrainer 初始化 (trainer/distillation.py)

在 `__init__` 中根据 `config.distribution_loss` 字段选择具体模型类：

```python
# trainer/distillation.py:61-68
if config.distribution_loss == "causvid":
    self.model = CausVid(config, device=self.device)
elif config.distribution_loss == "dmd":
    self.model = DMD(config, device=self.device)
elif config.distribution_loss == "sid":
    self.model = SiD(config, device=self.device)
```

### DMD vs SiD 配置差异

| 配置项 | DMD (`self_forcing_dmd.yaml`) | SiD (`self_forcing_sid.yaml`) |
|--------|------|------|
| `distribution_loss` | `"dmd"` | `"dmd"`（注意：SiD yaml 中目前也写的 `dmd`，需改为 `"sid"` 才会实例化 `SiD` 类）|
| `real_name` (teacher) | `Wan2.1-T2V-14B` (14B 大模型) | `Wan2.1-T2V-1.3B` (1.3B 小模型) |
| `lr_critic` | `4.0e-07` | `2.0e-06`（高 5 倍）|
| `weight_decay` | `0.01`（默认值） | `0.0` |
| `sid_alpha` | 无 | `1.0` |

共有配置：`denoising_step_list: [1000, 750, 500, 250]`、`num_frame_per_block: 3`、`dfake_gen_update_ratio: 5`、`ema_weight: 0.99`。

---

## 3. 模型类继承关系

```
BaseModel (model/base.py)
  ├── 初始化 generator, real_score, fake_score, text_encoder, vae, scheduler
  └── _get_timestep()：采样时间步
      │
      └── SelfForcingModel (model/base.py:98)
          ├── _run_generator()：执行 backward simulation 并切片最后 21 帧
          ├── _consistency_backward_simulation()：调用 training pipeline
          ├── _initialize_inference_pipeline()：延迟初始化 pipeline
          │
          ├── DMD (model/dmd.py)
          │   ├── _compute_kl_grad()：计算 KL 梯度
          │   ├── compute_distribution_matching_loss()：DMD 损失
          │   ├── generator_loss()
          │   └── critic_loss()
          │
          └── SiD (model/sid.py)
              ├── compute_distribution_matching_loss()：SiD 损失
              ├── generator_loss()
              └── critic_loss()
```

### BaseModel 初始化 (model/base.py:26-45)

```python
self.generator = WanDiffusionWrapper(is_causal=True)    # 因果模型，可训练
self.real_score = WanDiffusionWrapper(model_name=real_name, is_causal=False)  # 冻结
self.fake_score = WanDiffusionWrapper(model_name=fake_name, is_causal=False)  # 可训练
self.text_encoder = WanTextEncoder()   # 冻结
self.vae = WanVAEWrapper()             # 冻结
```

---

## 4. 核心训练循环

### ScoreDistillationTrainer.train() (trainer/distillation.py:312-389)

```
while True:
    TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)  # 默认每 5 步训 1 次 generator

    if TRAIN_GENERATOR:
        generator_optimizer.zero_grad()
        generator_loss = model.generator_loss(...)    ← 调用 DMD 或 SiD
        generator_loss.backward()
        generator.clip_grad_norm_(10.0)
        generator_optimizer.step()
        generator_ema.update(generator)               ← EMA 更新

    critic_optimizer.zero_grad()
    critic_loss = model.critic_loss(...)              ← 每步都训 critic
    critic_loss.backward()
    fake_score.clip_grad_norm_(10.0)
    critic_optimizer.step()

    step += 1
```

**更新比例**：generator : critic = 1 : 5（由 `dfake_gen_update_ratio=5` 控制）。

---

## 5. Self-Forcing Training Pipeline

### 核心：inference_with_trajectory() (pipeline/self_forcing_training.py:60)

这是 Self-Forcing 的精髓——在训练时模拟推理过程，用 KV cache 逐块生成视频。

#### 算法流程

```
输入: noise [B, F, C, H, W]，纯高斯噪声

1. 初始化 KV cache（30 层 × [B, cache_size, 12, 128]）

2. 如果有 initial_latent（I2V 场景），先过一遍 generator 写入 cache

3. 对每个 block 进行时空去噪：
   for block_index in all_blocks:
       noisy_input = noise 的对应切片

       # 随机选择一个退出点（用于反向传播）
       exit_flag = random_index in [0, len(denoising_steps))

       for index, timestep in enumerate(denoising_step_list):
           if index != exit_flag:
               # 不回传梯度，正常去噪
               with torch.no_grad():
                   denoised = generator(noisy_input, timestep, kv_cache)
                   noisy_input = add_noise(denoised, next_timestep)
           else:
               # 到达退出点，执行有梯度的前向
               if current_frame < start_gradient_frame:
                   with torch.no_grad():  # 前面的帧不需要梯度
                       denoised = generator(noisy_input, timestep, kv_cache)
               else:
                   denoised = generator(noisy_input, timestep, kv_cache)  # 有梯度
               break

       # 记录输出
       output[current_start:current_start+num_frames] = denoised

       # 重新运行 generator（timestep=0），更新 KV cache 供下一块使用
       with torch.no_grad():
           generator(denoised, timestep=context_noise, kv_cache)

       current_start += num_frames
```

#### 关键设计

1. **随机退出策略**：每个 block 随机选择在哪个去噪步骤退出并回传梯度，避免只优化某一特定步骤
2. **梯度裁剪**：只有最后 21 帧（训练目标帧）参与梯度计算，前面的帧全部 `no_grad`
3. **Context 更新**：每个 block 生成完后，以 `timestep=0`（或 `context_noise`）重新跑一遍 generator，将生成结果写入 KV cache 作为后续 block 的上下文
4. **分块结构**：默认 21 帧 = 7 blocks × 3 帧/block

---

## 6. DMD 损失详解

### 核心代码：model/dmd.py

#### generator_loss() 流程

```
1. _run_generator()
   └── _consistency_backward_simulation()
       └── inference_with_trajectory()   ← 自回归生成 21 帧（带梯度）

2. compute_distribution_matching_loss(生成的视频 x₀)
   ├── 采样随机时间步 t
   ├── 加噪：x_t = add_noise(x₀, noise, t)
   ├── _compute_kl_grad(x_t, x₀, t)
   │   ├── pred_fake = fake_score(x_t, t)           ← critic 预测
   │   ├── pred_real = real_score(x_t, t) + CFG      ← teacher 预测（带 CFG=3.0）
   │   ├── grad = pred_fake - pred_real               ← KL 梯度（DMD 论文 eq.7）
   │   └── grad = grad / |x₀ - pred_real|            ← 归一化（DMD 论文 eq.8）
   └── loss = 0.5 * MSE(x₀, (x₀ - grad).detach())
```

#### DMD 损失公式

来自 [DMD 论文](https://arxiv.org/abs/2311.18828) eq.7-8：

```
grad = (f_fake(x_t, t) - f_real(x_t, t)) / |x₀ - f_real(x_t, t)|

L_DMD = 0.5 * ||x₀ - sg(x₀ - grad)||²
```

其中 `sg` 表示 stop-gradient（`.detach()`），`f_fake` 和 `f_real` 分别是 fake_score 和 real_score 的 x₀ 预测。

#### critic_loss() 流程

```
1. _run_generator()（无梯度）→ 生成视频
2. 采样新的时间步 t'，对生成视频加噪
3. pred_fake = fake_score(noisy_generated, t')
4. loss = FlowPredLoss(pred_fake, generated_video)   ← 标准去噪损失
```

Critic 的目标是学会在 generator 生成的分布上做准确的去噪预测。

---

## 7. SiD 损失详解

### 核心代码：model/sid.py

SiD 与 DMD 的 `generator_loss()` 和 `critic_loss()` 结构相同，区别在于 `compute_distribution_matching_loss()` 中的损失公式。

#### SiD 损失公式

```python
# model/sid.py:128
sid_loss = (pred_real - pred_fake) * ((pred_real - x₀) - α * (pred_real - pred_fake))
```

展开：
```
L_SiD = (f_real - f_fake) · [(f_real - x₀) - α·(f_real - f_fake)]
```

其中 `α = sid_alpha`（默认 1.0）。

#### DMD vs SiD 损失对比

| 方面 | DMD | SiD |
|------|-----|-----|
| 损失形式 | `MSE(x₀, sg(x₀ - grad))` | `(f_real - f_fake) · [(f_real - x₀) - α·(f_real - f_fake)]` |
| 梯度计算 | 在 `torch.no_grad()` 内采样时间步 | 时间步采样不在 `no_grad` 内（梯度可流过） |
| fake_guidance_scale | 可配置（默认 0.0） | 不使用 |
| 归一化 | 在 `_compute_kl_grad` 内部做 | 单独用 `|x₀ - f_real|` 归一化 |
| Teacher 规模 | 14B（DMD 配置） | 1.3B（SiD 配置） |

---

## 8. 完整训练流程图

```
┌─────────────────────────────────────────────────────────┐
│                    train.py 入口                         │
│  加载 default_config.yaml + self_forcing_dmd.yaml       │
│  创建 ScoreDistillationTrainer                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             ScoreDistillationTrainer.__init__            │
│                                                         │
│  ① 根据 distribution_loss 创建 DMD / SiD 模型           │
│  ② FSDP 分布式包装 generator / real_score / fake_score  │
│  ③ 创建两个 AdamW 优化器                                │
│  ④ 加载文本 prompt 数据集                               │
│  ⑤ 初始化 EMA                                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 训练主循环 train()                        │
│                                                         │
│  while True:                                            │
│  ┌───────────────────────────────────────────┐          │
│  │  每 5 步: Generator Step                   │          │
│  │                                           │          │
│  │  ① text_encoder 编码 prompt               │          │
│  │  ② generator_loss():                      │          │
│  │     _run_generator():                     │          │
│  │       inference_with_trajectory()         │          │
│  │         逐块自回归生成 → 输出 21 帧        │          │
│  │     compute_distribution_matching_loss(): │          │
│  │       加噪 → 算 KL grad → MSE 损失       │          │
│  │  ③ backward + clip_grad + optimizer.step  │          │
│  │  ④ EMA 更新                               │          │
│  └───────────────────────────────────────────┘          │
│  ┌───────────────────────────────────────────┐          │
│  │  每步: Critic Step                         │          │
│  │                                           │          │
│  │  ① text_encoder 编码 prompt               │          │
│  │  ② critic_loss():                         │          │
│  │     _run_generator() [no_grad]            │          │
│  │     对生成视频加噪                         │          │
│  │     fake_score 预测 → 去噪损失             │          │
│  │  ③ backward + clip_grad + optimizer.step  │          │
│  └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 关键文件索引

| 文件 | 核心内容 |
|------|---------|
| `train.py` | 入口，配置加载与 trainer 路由 |
| `trainer/distillation.py` | `ScoreDistillationTrainer`：训练主循环、FSDP 包装、优化器、EMA |
| `model/base.py` | `BaseModel`：三模型初始化；`SelfForcingModel`：backward simulation 接口 |
| `model/dmd.py` | `DMD`：DMD 损失计算（`_compute_kl_grad`、`compute_distribution_matching_loss`） |
| `model/sid.py` | `SiD`：SiD 损失计算 |
| `pipeline/self_forcing_training.py` | `SelfForcingTrainingPipeline`：分块自回归推理模拟，KV cache 管理 |
| `utils/wan_wrapper.py` | `WanDiffusionWrapper`：模型包装层，flow → x₀ 转换 |
| `wan/modules/causal_model.py` | `CausalWanModel`：带 KV cache 的因果 Transformer |
| `utils/loss.py` | 去噪损失函数（FlowPredLoss 等），用于 critic 训练 |
| `configs/default_config.yaml` | 基础默认配置 |
| `configs/self_forcing_dmd.yaml` | DMD 训练配置 |
| `configs/self_forcing_sid.yaml` | SiD 训练配置 |

---

## 10. 关键概念总结

### 为什么是 data-free？

训练不需要真实视频。Generator 从纯噪声出发，通过 `inference_with_trajectory()` 自回归生成视频（backward simulation），然后用 teacher 模型的分布信号来校正 generator 的输出。这来自 DMD2 论文 Sec 4.5 的思想。

### Self-Forcing 解决了什么？

传统训练中，模型在 ground truth 上训练，但推理时在自己生成的帧上运行，导致 train-test distribution mismatch。Self-Forcing 在训练时就让模型在自己生成的帧上运行，从根源消除这个差异。

### KV Cache 的作用

CausalWanModel 使用 KV cache 实现分块自回归：每生成一个 block（3 帧），将其特征写入 cache，后续 block 的注意力可以看到前面所有已生成的内容。这使得显存消耗有界（由 `local_attn_size` 控制），可以扩展到长视频生成。
