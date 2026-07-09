# Adaptive World-Model-Guided ACT 计划

## 目标

当前 `ACT + AdaJEPA` 已经能通过 AdaJEPA latent transition objective 提升策略表现，但如果 AdaJEPA 只作为评估器或辅助损失，提升空间会比较有限。下一步的核心目标是把 AdaJEPA 从 evaluator 升级为 planner / corrector / recovery model。

推荐主线：

```text
Adaptive World-Model-Guided ACT

ACT 负责生成可行的专家动作块；
AdaJEPA 负责预测、筛选、修正和在线自适应。
```

最终推理流程从：

```text
当前观测 -> ACT 输出动作 chunk -> AdaJEPA 预测/评估 -> 执行 -> 更新 AdaJEPA
```

升级为：

```text
当前观测
    -> ACT 生成动作初值
    -> AdaJEPA 预测未来 latent state
    -> 根据目标距离、预测误差和不确定性修正动作
    -> 自适应选择执行 horizon
    -> 执行修正后的动作前缀
    -> 重新观测并在线更新世界模型
```

## 现有瓶颈判断

当前结构大概率是：

```text
obs_t -> ACT -> a_ACT[t:t+H] -> AdaJEPA latent prediction/evaluation -> execute
```

这类结构通常能带来稳定提升，但 AdaJEPA 只是从 ACT 给出的动作里评估或辅助训练，并不能主动创造更好的动作。对于拆装、插入、旋转、接触类任务，失败往往不是策略完全不会做，而是局部误差导致：

- 位姿偏一点；
- 接触力不合适；
- 插入角度不对；
- 摩擦和仿真有差异；
- 目标物体姿态有小偏差；
- 固定 chunk 在接触阶段执行太久。

因此下一步应该让 AdaJEPA 直接参与动作修正和执行 horizon 控制。

## 创新点 1：ACT + AdaJEPA Residual MPC

### 核心思想

ACT 先输出一个专家风格动作块：

```text
a_ACT = [a_t, a_{t+1}, ..., a_{t+H}]
```

不要直接执行，而是在 AdaJEPA latent dynamics 中优化一个小残差：

```text
a = a_ACT + delta_a
```

优化目标：

```text
min delta_a
    C_goal
  + lambda_bc * C_bc
  + lambda_smooth * C_smooth
  + lambda_uncertainty * C_uncertainty
  + lambda_force * C_force
```

其中：

```text
C_goal        = || z_pred_end - z_goal ||^2
C_bc          = || delta_a ||^2
C_smooth      = || a_t - a_{t-1} ||^2
C_uncertainty = predicted latent/action uncertainty
C_force       = contact force or force proxy penalty
```

解释：

- ACT 给出一个像专家的动作初值；
- AdaJEPA 预测这个动作执行后的 latent trajectory；
- 如果预测离目标或子目标有偏差，就优化一个小修正；
- residual 不能太大，避免动作离开专家分布。

### 第一版实现范围

第一版先不要做复杂 recovery policy，先实现可控的 residual correction：

```text
obs_t
    -> ACT predict_action_chunk
    -> AdaJEPA rollout_latents
    -> optimize delta_a for N steps
    -> clip residual
    -> output corrected chunk
```

建议先固定：

- `mpc_num_iters = 5`
- `mpc_lr = 1e-2`
- `mpc_residual_clip = 0.05`
- `mpc_bc_weight = 0.1`
- `mpc_smooth_weight = 0.05`

### 需要新增的配置

文件：

```text
src/lerobot/policies/actadajepa/configuration_actadajepa.py
```

建议新增：

```python
use_residual_mpc: bool = False
mpc_num_iters: int = 5
mpc_lr: float = 1e-2
mpc_horizon: int | None = None
mpc_residual_clip: float = 0.05

mpc_goal_weight: float = 1.0
mpc_bc_weight: float = 0.1
mpc_smooth_weight: float = 0.05
mpc_uncertainty_weight: float = 0.1
mpc_force_weight: float = 0.0
```

### 需要新增的模型接口

文件：

```text
src/lerobot/policies/actadajepa/modeling_actadajepa.py
```

现有接口已经有：

```python
encode_jepa_observation(batch)
predict_next_jepa_latent(batch, action)
jepa_prediction_loss(...)
```

Residual MPC 需要从 latent 开始递推，所以建议新增：

```python
def predict_next_jepa_latent_from_latent(self, latent: Tensor, action: Tensor) -> Tensor:
    action_latent = self.jepa_action_encoder(action)
    return self.jepa_predictor(torch.cat([latent, action_latent], dim=-1))


def rollout_jepa_latents(self, initial_latent: Tensor, actions: Tensor) -> Tensor:
    latents = []
    latent = initial_latent
    for step in range(actions.shape[1]):
        latent = self.predict_next_jepa_latent_from_latent(latent, actions[:, step])
        latents.append(latent)
    return torch.stack(latents, dim=1)
```

### 需要新增的 policy 接口

建议在 `ACTAdaJEPAPolicy` 中新增：

```python
def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
    act_actions = self.predict_action_chunk(batch)
    if not self.config.use_residual_mpc:
        return act_actions, {}
    corrected_actions, info = self.optimize_residual_actions(batch, act_actions)
    return corrected_actions, info
```

再新增：

```python
def optimize_residual_actions(
    self,
    batch: dict[str, Tensor],
    act_actions: Tensor,
) -> tuple[Tensor, dict]:
    ...
```

注意点：

- 只优化 `delta_a`，不更新模型参数；
- `act_actions` 作为 anchor；
- residual 需要 clamp；
- 优化过程允许对 action 求梯度，但模型参数应保持不更新；
- 第一版可先只用 latent goal cost + residual L2 + smoothness。

## 创新点 2：自适应 chunk 长度

### 核心思想

ACT 的固定 chunk 长度不适合所有阶段：

- 自由空间移动：可以执行长 chunk；
- 接近接触：应该短 chunk，高频重规划；
- 插入/旋转：应该更短、更谨慎；
- 预测误差或不确定性变大：马上 replan。

执行逻辑：

```text
ACT 输出 H 步动作
AdaJEPA 预测每一步 uncertainty / risk
找到超过阈值的位置 h
只执行前 h 步
重新观测并重新规划
```

### Horizon 选择规则

第一版用 heuristic 即可：

```text
h_exec = n_action_steps

如果 uncertainty[h] > uncertainty_threshold:
    h_exec = min(h_exec, h)

如果 prediction_error[h] > prediction_error_threshold:
    h_exec = min(h_exec, h)

如果检测到 contact / force spike:
    h_exec = min(h_exec, contact_replan_horizon)

h_exec = clamp(h_exec, min_execution_horizon, max_execution_horizon)
```

### 需要新增的配置

文件：

```text
src/lerobot/policies/actadajepa/configuration_actadajepa.py
```

建议新增：

```python
use_adaptive_horizon: bool = False
min_execution_horizon: int = 1
max_execution_horizon: int | None = None
uncertainty_threshold: float = 0.2
prediction_error_threshold: float = 0.2
contact_replan: bool = True
contact_replan_horizon: int = 1
```

### 替换固定 action queue 逻辑

当前 ACT 类似逻辑：

```python
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()
```

建议在 `ACTAdaJEPAPolicy.select_action` 中改成：

```python
if len(self._action_queue) == 0:
    actions, plan_info = self.plan_action_chunk(batch)
    execution_horizon = self.choose_execution_horizon(plan_info)
    actions = actions[:, :execution_horizon]
    self._action_queue.extend(actions.transpose(0, 1))

return self._action_queue.popleft()
```

新增：

```python
def choose_execution_horizon(self, plan_info: dict) -> int:
    ...
```

## 诊断与日志

需要记录以下指标，否则很难判断瓶颈：

- `jepa_loss`
- `jepa_pred_error`
- `rollout_error_mean`
- `rollout_error_final`
- `mpc_residual_norm`
- `mpc_goal_cost`
- `mpc_bc_cost`
- `mpc_smooth_cost`
- `mpc_uncertainty_cost`
- `adaptive_execution_horizon`
- `adaptive_replan_reason`

建议优先接入 wandb：

```text
wandb.log({
    "eval/mpc_residual_norm": ...,
    "eval/adaptive_execution_horizon": ...,
    "eval/jepa_rollout_error": ...,
})
```

## 实验设计

建议按 ablation 顺序跑：

| 编号 | 方法 | 目的 |
| --- | --- | --- |
| 1 | ACT | baseline |
| 2 | ACT + AdaJEPA loss | 验证 latent dynamics 辅助训练收益 |
| 3 | ACT + AdaJEPA evaluator/rerank | 验证只评估动作的上限 |
| 4 | ACT + AdaJEPA Residual MPC | 验证主动修正动作的收益 |
| 5 | ACT + Adaptive Horizon | 验证自适应执行长度收益 |
| 6 | ACT + Residual MPC + Adaptive Horizon | 验证完整方法 |

重点指标：

- success rate；
- average episode length；
- contact-stage failure rate；
- residual norm；
- average execution horizon；
- prediction error 和成功率的相关性；
- residual 是否导致动作抖动。

## 推荐实现顺序

### Step 1：补 AdaJEPA rollout 能力

- 新增 `predict_next_jepa_latent_from_latent`；
- 新增 `rollout_jepa_latents`；
- 保持现有训练 loss 不变。

### Step 2：实现 Residual MPC

- 新增 `use_residual_mpc` 配置；
- 新增 `optimize_residual_actions`；
- 第一版只优化 `delta_a`；
- 对 residual 做 clamp；
- 输出 corrected action chunk。

### Step 3：实现 Adaptive Horizon

- 新增 `use_adaptive_horizon` 配置；
- 新增 `choose_execution_horizon`；
- 将固定 `n_action_steps` queue 改成动态 horizon queue。

### Step 4：接日志和 ablation

- 加 plan info；
- 记录 residual、cost、horizon；
- 跑 ACT / ACTAdaJEPA / Residual MPC / Adaptive Horizon 对比。

### Step 5：再扩展失败恢复和子目标规划

在前两项跑通后，再加：

- subgoal latent predictor；
- failure detector；
- recovery action proposal；
- online AdaJEPA update buffer。

## 最小可行版本

第一版 MVP 只需要完成：

```text
1. AdaJEPA latent rollout
2. Residual action optimization
3. Adaptive execution horizon
4. 基础日志
```

推荐先不开启所有功能：

```bash
lerobot-train \
  --policy.type=actadajepa \
  --policy.use_residual_mpc=false \
  --policy.use_adaptive_horizon=false
```

训练稳定后，再评估时打开：

```bash
lerobot-eval \
  --policy.path=outputs/train/actadajepa_pusht_gpu4/checkpoints/last/pretrained_model \
  --policy.use_residual_mpc=true \
  --policy.use_adaptive_horizon=true
```

## 预期贡献表述

可以把方法贡献写成：

```text
We propose Adaptive World-Model-Guided ACT, a policy execution framework that augments action chunking transformers with an adaptive latent world model. Instead of using the world model only as an auxiliary training objective or action evaluator, the proposed method uses AdaJEPA to perform residual action correction and uncertainty-aware adaptive horizon selection during inference.
```

中文表述：

```text
本文提出 Adaptive World-Model-Guided ACT。该方法将 ACT 作为专家动作块生成器，将 AdaJEPA 作为 latent world model，在推理阶段进行 residual action correction 和 uncertainty-aware adaptive horizon selection，使世界模型从被动评估器升级为主动规划与修正模块。
```
