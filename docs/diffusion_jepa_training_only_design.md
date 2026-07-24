# Diffusion-JEPA：独立双编码器、训练期监督

## 结论

当前架构按 VLA-JEPA 图中的职责边界设计，但用 Diffusion Policy 替换 LLM/action head：

```text
                    ┌──────────── 策略分支 ────────────┐
observation history -> Vision Encoder -> Diffusion U-Net -> action chunk
                                      └-> latent action tokens
                                                    │
                    ┌──── 仅训练时运行的 JEPA 分支 ────┐
current/future obs -> JEPA Encoder -> state latents │
current state latent + latent action tokens         │
                    -> Latent World Model            │
                    -> predicted future state latent │
                    -> alignment + SIGReg loss       │
```

两个 encoder 是两个独立的模块实例，没有 backbone、adapter 或 projector 参数共享。
JEPA state 不融合进 Diffusion condition，也不用于 candidate rerank。

## 训练数据流

给定观测序列和专家动作：

```text
o_{t-n+1:t+H}, a_{t-n+1:t+H-1}
```

策略分支执行标准 Diffusion 训练：

```text
p = VisionEncoder(o_{t-n+1:t})
x_tau = add_noise(a, epsilon, tau)
epsilon_hat, h_diff = UNet(x_tau, tau, condition=p)
L_diffusion = mse(epsilon_hat, epsilon)
u = LatentActionHead(h_diff)
```

`h_diff` 取 U-Net decoder 在最终 action projection 之前的逐时刻隐藏特征，因此
`u_i` 与 action horizon 一一对齐。latent-action head 只在训练 JEPA loss 时执行。

JEPA 分支独立编码当前和未来观测：

```text
z_i = JEPAEncoder(o_i)
z_hat_{i+1} = WorldModel(z_<=i, u_<=i)
L_align = mse(z_hat_{i+1}, z_{i+1})
L_jepa = L_align + lambda_sigreg * SIGReg(z)
L_total = L_diffusion + lambda_jepa * L_jepa
```

World model 通过两条路径接收 latent action：

- 显式 latent-action token embedding，保证第一步训练就有梯度回到 Diffusion；
- 每个 Transformer block 的 action-conditioned AdaLN。

因此 alignment loss 的梯度路径为：

```text
L_align
  -> Latent World Model
  -> latent action tokens
  -> LatentActionHead
  -> Diffusion U-Net hidden features
  -> Vision Encoder condition
```

JEPA encoder 由 alignment loss 和 SIGReg 训练，但其参数不进入策略分支。

## 推理数据流

```text
observation history
  -> Vision Encoder
  -> Diffusion iterative denoising
  -> action chunk
```

推理不会调用：

- JEPA Encoder；
- latent-action head；
- Latent World Model；
- SIGReg；
- goal latent 或 candidate scoring。

所以部署延迟与普通 Diffusion Policy 保持同一条主路径。JEPA 相关权重可以在后续导出阶段
裁剪，但在训练 checkpoint 中保留以支持继续训练。

## 当前范围

当前代码实现图中的 robot-data finetuning 数据流。human-video pretraining 还需要一个
不依赖机器人 action/noisy action 输入的 latent-action query 路径，不能直接复用当前
U-Net denoising输入；在定义 human-video dataset 接口之前不将其伪装成已经支持。

## 关键配置

```text
jepa_latent_dim
jepa_action_latent_dim
jepa_prediction_horizon
jepa_world_model_loss_weight
jepa_sigreg_weight
jepa_loss_ramp_steps
```

旧版 `jepa_condition_residual_scale`、shared-encoder gradient scale 和 candidate-selection
字段仅为读取旧 config 保留，当前模型不会使用；`use_jepa_candidate_selection=true`
会直接报错。

## 验证标准

- Vision Encoder 与 JEPA Encoder 是不同模块、不同参数；
- alignment loss 对 latent-action head 产生非零梯度；
- 固定 U-Net 权重和 noise 时，推理动作与普通 Diffusion Policy 一致；
- action generation 不触发任何 JEPA 模块；
- JEPA predictor 保持因果性，未来 token 不影响较早预测；
- 保存与加载后训练步数和模型参数一致。
