# Diffusion + LeWorldModel JEPA 实现计划

## 1. 结论

可以整合，推荐采用下面的组合，而不是直接把 ACT-JEPA 的 predictor 接到 Diffusion 后面：

```text
ACT-JEPA 提供整合思路：
    同一个 observation encoder
        ├── 服务动作策略
        └── 服务未来表征预测
    两个目标端到端联合训练

LeWorldModel 提供 JEPA/world model 结构：
    z_t = encoder(o_t)
    z_hat_t+1 = causal_predictor(z_<=t, a_<=t)
    action 通过 AdaLN 注入 predictor
    prediction loss + SIGReg
    不使用 EMA、stop-gradient 或 target encoder

本项目中的动作策略：
    用 Diffusion 替换 ACT 的确定性 action decoder
```

最终得到的不是“Diffusion 和 JEPA 两个互不相关的模块”，而是：

```text
共享 observation encoder
    ├── Diffusion：生成多种可能的 action chunks
    └── LeWM-JEPA：预测每种 action 导致的未来 latent
```

这保留了 Diffusion 对多模态动作分布的建模能力，同时让共享表征学习动作相关的环境动力学。

## 2. 两篇论文分别采用什么

### 2.1 从 ACT-JEPA 采用

- observation encoder 同时服务策略学习和未来表征预测；
- action loss 与 JEPA loss 在同一训练阶段联合优化；
- 图像、proprioception 和任务信息先分别编码，再融合成共享 observation representation；
- JEPA 只在训练时也能发挥 representation regularization 的作用，推理不必强制运行 world model；
- action decoder 可以替换成 Diffusion 等更强的生成模型，ACT-JEPA 论文也明确提到了这一点。

### 2.2 不照搬 ACT-JEPA 的部分

ACT-JEPA 的 predictor 输入是当前 context 与 future mask tokens，没有输入 action。它学习的是：

```text
current observation -> likely future observation sequence
```

这在单一专家数据上可以学习典型演化，但不是可控动力学模型，无法回答：

```text
执行 action candidate A 和 B，分别会产生什么结果？
```

因此本方案不采用 ACT-JEPA 的无 action predictor，也不采用其 EMA target encoder。

### 2.3 从 LeWorldModel 采用

- 单一 online encoder；
- action-conditioned causal Transformer predictor；
- action 在每层通过零初始化的 AdaLN 注入；
- teacher-forcing 的 next-embedding MSE；
- SIGReg 防止 encoder 映射到常量；
- encoder、predictor 和 target embedding 全部端到端反向传播；
- 多步推理时在 latent space 自回归 rollout。

LeWorldModel 论文的默认 predictor 是 6 层、16 个 attention heads、10% dropout；latent 维度为 192，SIGReg 默认使用 1024 个随机方向、权重 0.1。第一版应保留这些目标与接口，但模型宽度可根据机器人数据规模和 NPU 性能缩小。

## 3. 本轮范围

本轮计划只包含：

- Phase 0：固化 Diffusion baseline；
- Phase 1：完成联合训练版 Diffusion-LeWM-JEPA；
- Phase 2：用 JEPA rollout 对 Diffusion candidates 排序。

明确不加入第三阶段，也不做：

- adaptive execution horizon；
- 卡点检测、恢复动作或 residual MPC；
- JEPA gradient guidance 注入 diffusion denoising；
- test-time online adaptation；
- 执行过程中的闭环重规划扩展。

Phase 2 选出 action chunk 后，仍沿用原始 Diffusion Policy 的固定 action queue 和 `n_action_steps`。

## 4. 推荐架构

### 4.1 总体数据流

```text
训练输入
    observation sequence: o_{t-n+1}, ..., o_t, o_{t+1}, ..., o_{t+H}
    expert action chunk:   a_t, a_{t+1}, ..., a_{t+H-1}

每帧共享编码
    RGB camera(s) -> visual backbone -> visual features
    robot state   -> state projection
    env/task info -> optional projection
    concat -> 原始控制特征 r_i
        ├── 保留为 Diffusion baseline condition
        └── fusion MLP + projector -> JEPA latent z_i

                       ┌──────────────────────────────────────────┐
r_{t-n+1:t} ---------->│ condition = r + 0.1 * ZeroLinear(z)      │
z_{t-n+1:t} ---------->│ -> conditional 1D U-Net                  │
expert action chunk -->│ -> denoising loss L_diffusion            │
                       └──────────────────────────────────────────┘

z sequence + clean actions
    -> causal Transformer predictor
    -> actions injected by AdaLN at every block
    -> z_hat_{i+1}
    -> next-embedding loss L_pred

all z_i
    -> step-wise SIGReg
    -> L_sigreg
```

### 4.2 Shared observation encoder

推荐把共享 encoder 定义为逐时间步的双输出编码器：

```text
(r_i, z_i) = SharedObservationEncoder(
    images_i,
    robot_state_i,
    optional_environment_state_i,
    optional_task_i,
)
```

第一版继续使用 LeRobot Diffusion Policy 的 `DiffusionRgbEncoder`，原因是：

- 能与当前 Diffusion baseline 做公平比较；
- ACT-JEPA 本身使用 ResNet-18；
- LeWorldModel 的实验表明其训练目标并不依赖 ViT，ResNet-18 也可工作；
- 避免同时更换 policy、world model 和视觉 backbone，导致无法判断收益来源。

后续消融再比较 ViT-Tiny，而不是在 MVP 中强制替换。

建议融合结构：

```text
camera features ─┐
robot state -> Linear + SiLU
env/task feature -> Linear + SiLU（如存在）
                 └→ concat -> r_i

r_i -> Linear -> SiLU -> LayerNorm -> Linear -> step-wise BatchNorm -> z_i
```

初始配置：

```text
jepa_latent_dim = 192
```

LeWorldModel 在 ViT 的最终 LayerNorm 后增加 1 层 MLP + BatchNorm，使 SIGReg 能有效塑造输出分布。当前实现也保留独立 projector，并只对 `z_i` 计算 SIGReg。原始控制特征 `r_i` 不受 SIGReg 直接约束，避免丢失接近任务终点时所需的精细位置与小动作信息。

### 4.3 Diffusion 分支

当前 Diffusion Policy 使用最近 `n_obs_steps` 帧的原始特征作为 U-Net global condition。新模型在保留这条路径的基础上增加零初始化 JEPA residual：

```text
r_history = [r_{t-n+1}, ..., r_t]
z_history = [z_{t-n+1}, ..., z_t]

jepa_residual = ZeroInitLinear(z_history)
condition = r_history + residual_scale * jepa_residual
global_cond = flatten(condition)
```

`jepa_condition_residual` 的权重和偏置都初始化为零，默认 `residual_scale=0.1`。因此训练开始时：

```text
condition = r_history
```

即严格恢复原始 Diffusion condition。这样：

- `DiffusionConditionalUnet1d.global_cond_dim` 不变；
- U-Net 结构与输入语义在初始化时都与纯 Diffusion baseline 一致；
- 可以单独加载已有 U-Net 权重；
- Diffusion loss 始终可以沿原始 `r_i` 路径训练视觉 backbone；
- JEPA residual 只能逐渐增加信息，不能替代整个控制 condition；
- 将 `jepa_condition_residual_scale=0` 可得到严格的原始 condition 消融。

Diffusion 仍使用 clean action trajectory 加噪后进行 epsilon 或 sample prediction，不把 JEPA latent loss 混入噪声调度器。

### 4.4 LeWorldModel predictor

predictor 采用 causal Transformer，而不是旧计划中的 recurrent MLP：

```text
input tokens:      z_t, z_{t+1}, ..., z_{t+H-1}
aligned actions:   a_t, a_{t+1}, ..., a_{t+H-1}
causal attention:  token i 只能访问 <= i 的 latent
output:            z_hat_{i+1}
```

每个 Transformer block 使用 action-conditioned AdaLN：

```text
scale_i, shift_i, gate_i = action_mlp(a_i)
x_i = AdaLN(x_i; scale_i, shift_i)
```

action conditioning projection 采用零初始化，使训练开始时 predictor 先建立稳定 latent dynamics，再逐渐利用 action。

MVP 建议：

```text
jepa_predictor_layers = 4
jepa_predictor_heads = 6
jepa_predictor_dropout = 0.1
jepa_latent_dim = 192
```

论文默认 6 层、16 heads 可作为较大模型消融。`192 / 6 = 32`，MVP 的 head 维度合法，也更适合先验证 NPU 吞吐。

### 4.5 不使用 target encoder

本方案必须删除以下 ACT-JEPA 风格结构：

```text
EMA target encoder
stop_gradient(z_target)
target encoder update()
online/target crop synchronization
```

训练目标中的 `z_{t+1}` 来自同一个 shared encoder，并且梯度通过预测值和目标值两侧传播：

```text
z_{t+1} = encoder(o_{t+1})
z_hat_{t+1} = predictor(z_<=t, a_<=t)
L_pred = MSE(z_hat_{t+1}, z_{t+1})
```

避免 collapse 的责任由 SIGReg 承担。这是该方案与 ACT-JEPA 最本质的结构区别。

## 5. 训练目标

### 5.1 总损失

推荐写成两层权重：

```text
L_LeWM = L_pred + lambda_sigreg * L_SIGReg

L_total = L_diffusion + lambda_world_model * L_LeWM
```

其中：

```text
lambda_sigreg = 0.1        # LeWorldModel 论文默认值
lambda_world_model = 0.02  # 保守起始值，降低精细控制梯度冲突
sigreg_num_projections = 1024
```

`lambda_world_model` 是 Diffusion 与 world-model 两种任务之间新增的尺度平衡项，不应与 LeWorldModel 内部的 `lambda_sigreg` 混为一个参数。

### 5.2 Teacher-forcing next-embedding loss

对从当前时刻开始的训练序列一次编码。最近的历史观测只提供给 Diffusion condition；LeWorldModel 序列从当前 `z_t` 开始，后续真实 latent 用于 teacher forcing：

```text
z = encoder(observations_t_to_t_plus_H)          # (B, H+1, D)
z_hat_next = predictor(z[:, :-1], clean_actions) # (B, H, D)
```

损失为：

```text
L_pred = masked_mean(
    ||z_hat_next - z[:, 1:]||²
)
```

训练时使用真实的历史 latent，即 teacher forcing；Phase 2 对候选动作评分时才使用 predicted latent 自回归 rollout。

### 5.3 SIGReg

SIGReg 对每个时间步跨 batch 的 embeddings 做随机一维投影，再用 Epps-Pulley statistic 逼近标准高斯：

```text
for each time step i:
    L_sigreg_i = SIGReg(z[:, i, :])
L_SIGReg = mean_i(L_sigreg_i)
```

实现要求：

- 随机方向必须在当前 device 上生成并单位归一化；
- FP16/BF16 训练时，Epps-Pulley 积分与特征函数计算优先转 FP32；
- padding 时间步不能进入统计；
- 小 batch 时 SIGReg 估计会抖动，应优先保证有效样本量；
- 分布式训练需决定是否 all-gather embeddings，第一版若只使用 local batch，日志中必须明确；
- 记录 feature std、effective rank 和 pairwise distance，验证不是“loss 下降但 latent 塌缩”。

若 1024 个方向在昇腾 NPU 上开销过高，可测试 `256 / 512 / 1024`。论文认为方向数量对性能不敏感，但最终选择必须由本项目吞吐与消融决定。

### 5.4 World-model loss ramp

ACT-JEPA 强调联合训练，因此不先单独预训练再冻结 encoder。为降低共享 encoder 初期的梯度冲突，可以只对权重做短 ramp：

```text
0 ~ 20K steps:
    lambda_world_model 从 0 线性增加到 0.02

20K+ steps:
    完整联合训练，所有模块保持可学习
```

同时测试无 ramp 的联合训练版本。最终依据 encoder gradient cosine、Diffusion loss 和成功率选择，而不是默认认为 ramp 必须存在。

## 6. 数据与时间对齐

当前默认 Diffusion 配置为：

```text
n_obs_steps = 2
horizon = 16
n_action_steps = 8
observation offsets = [-1, 0]
action offsets = [-1, 0, 1, ..., 14]
```

当前时刻 `t` 对应 action tensor 中：

```python
action_start = n_obs_steps - 1
```

因此 world model 的第一步 action 必须是：

```python
a_t = batch[ACTION][:, action_start]
```

不能使用索引 0，否则默认配置下拿到的是 `a_{t-1}`。

若 `jepa_prediction_horizon = 4`，训练至少需要：

```text
observations: [-1, 0, 1, 2, 3, 4]
JEPA actions: [0, 1, 2, 3] 对应 a_t ... a_t+3
```

注意 Diffusion 的 action tensor 仍保留 `[-1, 0, ..., 14]`，JEPA 只是从 `action_start` 处截取干净动作。

valid mask 必须同时处理：

- episode boundary；
- observation padding；
- action padding；
- 不完整的最后一个 prediction step。

## 7. 图像增强

当前 `DiffusionRgbEncoder` 在模块内部执行 random crop。世界模型学习时间变化时，如果同一轨迹各帧使用独立随机 crop，会把 crop 抖动误认为环境动力学。

MVP 二选一：

1. 先设置 `crop_is_random=false`，验证完整链路；
2. 把 crop 移到 sequence augmentation，对同一条时间序列的所有帧共享 crop 参数。

推荐先做 1，再实现 2。由于本方案只有一个 encoder，不再存在 online/target 两个 encoder 的 crop 对齐问题，但仍然存在跨时间步一致性问题。

## 8. Phase 2：Diffusion candidates + JEPA rollout

### 8.1 候选生成

```text
current observation history
    -> K 组 diffusion noise
    -> K 条 action candidates
    -> shape: (B, K, horizon, action_dim)
```

建议第一版：

```text
num_candidates = 8
candidate_rollout_horizon = 4
```

### 8.2 候选结果预测

对每条 candidate，从当前 latent 开始自回归：

```text
z_hat_t+1 = predictor(z_history, candidate_action_t)
append z_hat_t+1
z_hat_t+2 = predictor(updated_history, candidate_action_t+1)
...
```

所有 `B * K` 候选应合并为一个 batch rollout，避免 Python 按候选循环。

### 8.3 必须有任务目标才能排序

world model 只能预测“动作会导致什么”，不能凭空判断“这个结果是否更好”。因此 Phase 2 必须至少提供一种 scorer：

1. goal observation latent distance；
2. 数据集可监督的 progress/value head；
3. reward/success model；
4. 环境提供的任务几何目标。

最贴近 LeWorldModel 的方式是 goal observation：

```text
z_goal = encoder(goal_observation)
goal_cost = ||goal_projector(z_pred_final) - goal_projector(z_goal)||²
```

候选总代价可以为：

```text
J = w_goal * goal_cost
  + w_smooth * action_smoothness
  + w_boundary * action_limit_penalty
```

在没有 goal/progress/reward 的情况下，只允许记录预测不确定性和动作平滑性作为诊断，不能宣称 JEPA 已经具有任务规划能力，也不默认开启 candidate selection。

### 8.4 与 LeWorldModel CEM 的关系

LeWorldModel 从随机 action sequence 出发，用 CEM 迭代优化。这里不直接照搬 CEM，而是让 Diffusion 作为数据驱动的 proposal distribution：

```text
LeWorldModel：随机 candidates -> CEM 多轮优化
本方案：      Diffusion expert-like candidates -> JEPA 一次批量排序
```

对高维机器人 action，这样更容易得到落在专家数据分布附近的候选，也更符合已有 imitation learning 数据。CEM refinement 不纳入当前范围。

## 9. 配置草案

```python
@PreTrainedConfig.register_subclass("diffusion_jepa")
@dataclass
class DiffusionJEPAConfig(DiffusionConfig):
    jepa_latent_dim: int = 192
    jepa_predictor_layers: int = 4
    jepa_predictor_heads: int = 6
    jepa_predictor_dropout: float = 0.1

    jepa_prediction_horizon: int = 4
    jepa_world_model_loss_weight: float = 0.02
    jepa_sigreg_weight: float = 0.1
    jepa_sigreg_num_projections: int = 512
    jepa_loss_ramp_steps: int = 20_000
    jepa_condition_residual_scale: float = 0.1

    use_jepa_candidate_selection: bool = False
    jepa_num_action_candidates: int = 8
    jepa_candidate_horizon: int = 4
```

校验规则：

- `jepa_latent_dim % jepa_predictor_heads == 0`；
- prediction/candidate horizon 大于 0，且不越过 Diffusion action horizon；
- loss weight 非负；
- SIGReg projection 数大于 0；
- 开启 candidate selection 时 candidate 数至少为 2；
- 开启 candidate selection 时必须配置有效的 goal/progress scorer。

配置中不再出现：

```text
jepa_target_ema_decay
jepa_variance_loss_weight
jepa_latent_std_floor
```

## 10. 代码组织

新增独立策略，不修改纯 Diffusion baseline 的行为：

```text
src/lerobot/policies/diffusion_jepa/
├── __init__.py
├── configuration_diffusion_jepa.py
├── modeling_diffusion_jepa.py
├── modeling_lewm_predictor.py
├── sigreg.py
└── processor_diffusion_jepa.py
```

职责：

```text
configuration_diffusion_jepa.py
    配置、delta indices、参数校验

modeling_diffusion_jepa.py
    DiffusionJEPAPolicy
    SharedObservationEncoder
    diffusion loss + LeWM loss
    candidate generation / rollout / scoring

modeling_lewm_predictor.py
    causal Transformer
    zero-init action AdaLN
    teacher forcing 与 autoregressive rollout

sigreg.py
    random projections
    Epps-Pulley statistic
    padding/device/dtype 处理

processor_diffusion_jepa.py
    复用 Diffusion normalization、device pipeline
```

接入位置：

```text
src/lerobot/policies/factory.py
src/lerobot/policies/__init__.py
src/lerobot/__init__.py
tests/policies/test_diffusion_jepa.py
```

## 11. 分阶段任务与验收

### Phase 0：Diffusion baseline

- 固定 dataset、seed、batch size 和评估 episode seeds；
- 训练当前纯 Diffusion Policy；
- 保存成功率、reward、动作多样性、延迟和 checkpoint；
- 确认现有动作卡住问题是否在 baseline 中复现。

验收：后续实验除目标改动外使用完全相同的训练和评估协议。

### Phase 1：联合训练 MVP

- 新建 `diffusion_jepa` 策略与配置；
- 实现逐帧 shared observation encoder；
- 同一次 backbone 前向同时生成原始控制特征与 JEPA latent；
- Diffusion 使用原始 condition 加 zero-init JEPA residual；
- 实现 causal Transformer predictor；
- 在每个 predictor block 中实现 zero-init action AdaLN；
- 实现 teacher-forcing next-embedding MSE；
- 实现论文一致的 step-wise SIGReg；
- 处理 action、observation 和 padding 对齐；
- 推理阶段先保持原始 Diffusion 行为。

验收：

- residual adapter 初始化时，global condition 与原始 Diffusion 严格一致；
- `jepa_condition_residual_scale=0` 时始终使用原始 Diffusion condition；
- shared encoder 同时收到 Diffusion、prediction 和 SIGReg 梯度；
- target embedding 一侧也有梯度，工程中不存在 EMA target encoder；
- 改变 clean action 会显著改变 predicted next latent；
- latent std/effective rank 不塌缩；
- train-only 成功率不显著低于纯 Diffusion baseline。

### Phase 2：候选采样与排序

- 一次生成 K 条 Diffusion action trajectories；
- batch 化执行 LeWM autoregressive rollout；
- 接入 goal latent 或可验证的 progress scorer；
- 实现 smoothness 与 action boundary 辅助代价；
- 选中 candidate 后沿用固定 action queue 执行。

验收：

- 固定 noise 时 candidates 与选择结果可复现；
- 不同 noise 产生有区分度的动作；
- 同一 latent 下改变 action 会改变 rollout；
- 合成轨迹测试中 scorer 能选出更接近 goal 的候选；
- candidate selection 的任务成功率不低于 Phase 1；
- 到此结束，不进入第三阶段。

## 12. 必须实现的测试

### 时间与 shape

- observation offsets 覆盖 `[-n_obs_steps+1, ..., H]`；
- `action_start = n_obs_steps - 1`；
- predictor 输出 `(B, T-1, D)`；
- candidate rollout 输出 `(B, K, H, D)`；
- episode padding 不进入 prediction loss 或 SIGReg。

### 因果性与 action conditioning

- 修改未来 latent 不影响较早 predictor 输出；
- 相同 latent、不同 action 得到不同 next latent；
- zero-init 时 action branch 初始影响接近 0；
- 训练后 AdaLN action parameters 获得非零梯度；
- autoregressive rollout 不偷看真实 future observations。

### 防塌缩

- 常量 embedding 的 SIGReg loss 明显高于近似标准高斯 embedding；
- SIGReg 对 encoder/projector 有有限且非零梯度；
- batch 太小或全 padding 时给出明确错误/跳过，而不是 NaN；
- FP32、FP16/BF16 路径数值有限。

### Diffusion 与 checkpoint

- Diffusion 输出 shape 与原策略一致；
- action queue、`reset()` 和 normalization 行为一致；
- 纯 Diffusion U-Net 权重可按设计加载；
- save/load 后固定 noise 得到相同输出。

> **Checkpoint 迁移说明：** 当前 residual 结构改变了 condition 的语义，不能继续训练旧版“JEPA latent
> 硬替换 Diffusion condition”的 checkpoint。旧权重中的 `diffusion_condition_projection` 不会加载到新的
> `jepa_condition_residual`；请新建 `output_dir` 从头训练，避免把旧 optimizer state 一并恢复。

### 昇腾 NPU

- CPU smoke test；
- Ascend NPU forward/backward smoke test；
- AdaLN、causal attention、随机投影与复数替代实现无 device mismatch；
- Epps-Pulley 实现不依赖 NPU 不支持的 complex op，优先用 `sin/cos` 实数形式；
- 第一版关闭 `torch.compile`，功能验证后再做图编译和算子融合；
- 记录 Diffusion、predictor、SIGReg 各自耗时与峰值显存。

## 13. 训练日志

至少记录：

```text
diffusion_loss
lewm_prediction_loss
sigreg_loss
weighted_world_model_loss
prediction_loss_step_1 ... step_H

latent_feature_std
latent_effective_rank
latent_pairwise_distance
predictor_action_sensitivity

encoder_grad_norm_from_diffusion
encoder_grad_norm_from_prediction
encoder_grad_norm_from_sigreg
diffusion_world_model_grad_cosine

npu_step_time_ms
npu_peak_memory_mb
```

Phase 2 另记录：

```text
candidate_action_diversity
candidate_goal_costs
selected_score_margin
world_model_rollout_error
candidate_generation_latency_ms
candidate_scoring_latency_ms
success_rate
```

## 14. 消融矩阵

| 实验 | Shared encoder | LeWM prediction | SIGReg | Candidate selection |
|---|---:|---:|---:|---:|
| 原始 Diffusion | 否 | 否 | 否 | 否 |
| Shared encoder only | 是 | 否 | 否 | 否 |
| Diffusion + prediction，无 SIGReg | 是 | 是 | 否 | 否 |
| Diffusion-LeWM-JEPA | 是 | 是 | 是 | 否 |
| Diffusion-LeWM-JEPA + candidates | 是 | 是 | 是 | 是 |

关键超参数：

```text
lambda_world_model = 0 / 0.01 / 0.02 / 0.05 / 0.1
lambda_sigreg = 0.03 / 0.1 / 0.3
condition_residual_scale = 0 / 0.1 / 1.0
prediction_horizon = 1 / 2 / 4 / 8
SIGReg projections = 256 / 512 / 1024
predictor layers = 2 / 4 / 6
num_candidates = 1 / 4 / 8 / 16
```

必须用相同 eval seeds 报告均值与置信区间。只比较训练 loss 不能证明策略变好。

## 15. 首轮建议配置

```text
jepa_latent_dim = 192
jepa_predictor_layers = 4
jepa_predictor_heads = 6
jepa_predictor_dropout = 0.1
jepa_prediction_horizon = 4

jepa_world_model_loss_weight = 0.02
jepa_sigreg_weight = 0.1
jepa_sigreg_num_projections = 512   # 先做 NPU 性能验证
jepa_loss_ramp_steps = 20000
jepa_condition_residual_scale = 0.1

use_jepa_candidate_selection = false
jepa_num_action_candidates = 8
jepa_candidate_horizon = 4
crop_is_random = false
```

先验证 512 projections，随后用论文默认 1024 做对照。如果二者任务表现相当，NPU 部署选择更快的配置。

## 16. 成功标准

### Phase 1 成功

- Diffusion 与 world-model losses 均稳定下降；
- action sensitivity 明显非零；
- latent 没有 collapse；
- 成功率至少不低于纯 Diffusion baseline；
- 多模态动作不再被确定性 action decoder 平均化。

### Phase 2 成功

- JEPA rollout error 随训练下降；
- candidate scorer 选择的预测结果更接近任务目标；
- 固定评估协议下成功率稳定高于 Phase 1 和纯 Diffusion；
- NPU 延迟与显存满足部署要求。

## 17. 最终原则

```text
Diffusion：提出多种数据分布内的动作方案
LeWM-JEPA：预测每种方案将导致的未来
Goal/progress scorer：判断哪种未来更符合任务目标
Shared encoder：通过两种训练信号学习既能控制又懂动力学的表征
```

三者不能混为一个 loss。尤其是 JEPA prediction error 小，只代表未来更可预测，不自动等于任务成功率更高；Phase 2 是否有效最终取决于 world model 精度和可验证的任务 scorer。
