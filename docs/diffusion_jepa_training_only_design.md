# Diffusion-JEPA：VLA-JEPA 对齐结构

## 模型边界

Diffusion-JEPA 使用两个完全独立的视觉编码器：

```text
策略历史观测
  -> Diffusion Vision Encoder（可训练）
  -> learnable latent-action queries
       ├-> Diffusion U-Net condition -> action chunk
       └-> token-level latent world model

当前及未来视频
  -> V-JEPA2 Encoder（预训练、冻结、仅训练时运行）
  -> multi-view world-state patch tokens
  -> token-level latent world model target
```

V-JEPA2 与 Diffusion Vision Encoder 不共享 backbone、adapter 或 projector。
V-JEPA2 只编码 RGB 视频，不编码机器人 state。
预处理器会在 Diffusion 数据集归一化之前保留一份原始 RGB：策略编码器使用原有
Diffusion-normalized 图像，V-JEPA2 使用原始 `[0, 1]` 图像后再应用其 ImageNet
预处理。

## 训练数据流

默认加载连续 8 帧，从当前时刻开始交给 tubelet size 为 2 的 V-JEPA2：

```text
8 video frames
  -> 4 world-state time steps
  -> 3 state transitions
```

每个状态保留全部空间 patch token。多相机特征按照 VLA-JEPA Equation (1)
沿 embedding dimension 拼接，不做全局池化。

策略视觉特征与 learnable queries 通过 cross-attention 生成：

```text
z: [B, 3 transitions, K latent tokens, D]
```

同一组 `z` 一方面汇聚后加入 Diffusion U-Net 的 global condition，另一方面与
V-JEPA2 state patch tokens 在 world model 中按时间块交错：

```text
[z_t^1 ... z_t^K, s_t^1 ... s_t^P]
```

同一时间块内为全注意力；不同时间块之间严格因果。预测器采用 teacher forcing：

```text
input  = states[:, :-1]
target = stop_gradient(states[:, 1:])
```

总损失与论文 Equation (9) 对齐：

```text
L = L_diffusion + beta * L1(predicted_next_states, target_next_states)
```

默认 `beta=0.1`。由于 target 来自冻结的预训练 V-JEPA2，不使用 SIGReg、EMA
target encoder 或防坍塌 projector。

## 推理数据流

```text
observation history
  -> Diffusion Vision Encoder
  -> latent-action queries
  -> Diffusion iterative denoising
  -> action chunk
```

推理不会调用 V-JEPA2 或 latent world model。latent-action query 模块属于
Diffusion Policy 本身，并在训练与推理中保持相同的数据流。
保存策略 checkpoint 时不会重复写入冻结的 V-JEPA2 权重；加载时按
`jepa_encoder_name_or_path` 重新读取官方 checkpoint。

## 关键配置

```text
jepa_encoder_name_or_path
jepa_num_frames = 8
jepa_prediction_horizon = 3
jepa_action_tokens_per_transition = 8
jepa_action_latent_dim = 1024
jepa_latent_dim = 1024
jepa_predictor_layers = 12
jepa_predictor_heads = 8
jepa_world_model_loss_weight = 0.1
```

`jepa_prediction_horizon` 必须等于
`jepa_num_frames / vjepa2_tubelet_size - 1`。

## 当前范围

当前策略 forward 实现带机器人动作标签的联合训练。latent-action queries 已不依赖
noisy action，因此模型结构具备 human-video world-model pretraining 的条件；但接入
Something-Something-v2 等无动作视频，还需要独立的视频 dataset、无 state 的视觉输入
接口及 co-training dataloader。

## 验证标准

- V-JEPA2 始终为 eval 且所有参数 `requires_grad=False`；
- 多相机 V-JEPA2 输出保持 patch token，并沿 embedding dimension 拼接；
- optimizer 不包含 V-JEPA2 参数；
- world-model L1 对 policy vision encoder 与 latent-action queries 产生梯度；
- 修改未来视频不影响 policy latent actions；
- 修改未来 world-model time block 不影响过去预测；
- action generation 不触发 V-JEPA2 或 world-model predictor。
