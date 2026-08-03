# 两阶段 ACT 训练使用说明（当前版本）

本文档说明当前仓库里可直接使用的训练流程：
- 阶段 0：将三种模具数据合并到一个数据集
- 阶段 1：三模具融合训练（co-training）
- 阶段 2：按模具独立头部微调（head-only finetune）

当前配置还会将三种模具的任务 ID 以 one-hot 形式拼接到 `observation.state`：

- 模具 A：`[1, 0, 0]`
- 模具 B：`[0, 1, 0]`
- 模具 C：`[0, 0, 1]`

`task_id_num_classes: 0` 是默认值，表示完全关闭任务条件并保持原始 ACT 行为。

适用范围：
- 已在本仓库中新增 `head_only_finetune` 开关
- 已提供 `examples/configs/two_stage_act` 下的配置模板

## 1. 环境与目录

建议在仓库目录执行所有命令：

```bash
cd /home/jing/rebot_lerobot/lerobot
```

建议使用当前已验证的 Python 环境：

```bash
/home/jing/miniforge3/envs/lerobot/bin/python
```

## 2. 配置文件清单

已提供以下模板文件：

- `examples/configs/two_stage_act/merge_three_molds.json`
- `examples/configs/two_stage_act/stage1_multimold_cotraining.json`
- `examples/configs/two_stage_act/stage2_moldA_head_finetune.json`
- `examples/configs/two_stage_act/stage2_moldB_head_finetune.json`
- `examples/configs/two_stage_act/stage2_moldC_head_finetune.json`

## 3. 阶段 0：三种模具数据合并

### 3.1 先修改合并配置

编辑：

- `examples/configs/two_stage_act/merge_three_molds.json`

需要替换：

- `repo_id`：合并后数据集名称（输出）
- `operation.repo_ids`：三个源数据集（模具 A/B/C）

示例（仅示意）：

```json
{
  "repo_id": "yourname/molds_merged",
  "operation": {
    "type": "merge",
    "repo_ids": [
      "yourname/mold_a",
      "yourname/mold_b",
      "yourname/mold_c"
    ]
  }
}
```

### 3.2 执行合并

```bash
/home/jing/miniforge3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_edit_dataset \
  --config_path examples/configs/two_stage_act/merge_three_molds.json
```

### 3.3 合并前必须一致的条件

三个数据集必须一致：

- `fps`
- `robot_type`
- `features`（观测/动作字段结构）

不一致会在合并校验时报错。

## 4. 阶段 1：三模具融合训练

### 4.1 修改阶段 1 配置

编辑：

- `examples/configs/two_stage_act/stage1_multimold_cotraining.json`

至少修改：

- `dataset.repo_id`：改为合并后的数据集
- `policy.device`：根据机器设置 `cuda`/`cpu`
- `steps`、`batch_size`：按资源调整

关键项说明：

- `dataset.episodes: null` 表示使用全部 episode（融合训练）
- `policy.head_only_finetune: false`（阶段 1 必须是 false）
- `policy.task_id_num_classes: 3`：启用三分类任务 one-hot
- `policy.task_id_override: null`：阶段 1 使用数据集 batch 中的 `task_index`

合并前请确保三份数据的 task 名称互不相同。合并工具按 task 名称生成 `task_index`；如果三份数据使用相同任务描述，它们会被当作同一个任务。

### 4.2 启动阶段 1

```bash
lerobot-train --config_path examples/configs/two_stage_act/stage1_multimold_cotraining.json
```

### 4.3 阶段 1 输出（供阶段 2 使用）

默认使用：

```text
outputs/train/taskbook_stage1_multimold/checkpoints/last/pretrained_model
```

## 5. 阶段 2：按模具独立微调（头部）

阶段 2 每个模具单独跑一次，分别产出模型。

### 5.1 修改阶段 2 配置（A/B/C 分别改）

编辑：

- `examples/configs/two_stage_act/stage2_moldA_head_finetune.json`
- `examples/configs/two_stage_act/stage2_moldB_head_finetune.json`
- `examples/configs/two_stage_act/stage2_moldC_head_finetune.json`

至少修改：

- `dataset.repo_id`：合并后的数据集
- `dataset.episodes`：该模具对应的 episode 列表
- `policy.pretrained_path`：阶段 1 的输出路径

关键项说明：

- `policy.head_only_finetune: true`
  - 只训练 `decoder + decoder_pos_embed + action_head`
  - 底层编码部分冻结
- `policy.task_id_num_classes: 3`
- `policy.task_id_override`：模具 A/B/C 分别设置为 `0/1/2`

阶段 2 使用 override 后，即使单模具数据集内部的 `task_index` 都是 0，也会输入正确的模具任务 ID。部署时加载对应专属 checkpoint，override 会随模型配置一起保存。

### 5.2 依次启动阶段 2

```bash
lerobot-train --config_path examples/configs/two_stage_act/stage2_moldA_head_finetune.json
lerobot-train --config_path examples/configs/two_stage_act/stage2_moldB_head_finetune.json
lerobot-train --config_path examples/configs/two_stage_act/stage2_moldC_head_finetune.json
```

## 6. 快速检查与常见问题

### 6.1 检查 JSON 配置语法

```bash
/home/jing/miniforge3/envs/lerobot/bin/python -m json.tool examples/configs/two_stage_act/stage1_multimold_cotraining.json >/dev/null && echo OK
```

### 6.2 训练时提示设备不可用

- 将 `policy.device` 改为当前可用设备（如 `cpu`）
- 关闭 `use_amp`（设为 `false`）

### 6.3 合并时报 features 不一致

通常是三个数据集字段结构不同（如相机键名不同）。
先统一三份数据的字段，再执行合并。

### 6.4 阶段 2 无法加载 pretrained_path

确认阶段 1 已完成并存在：

```text
outputs/train/taskbook_stage1_multimold/checkpoints/last/pretrained_model
```

## 7. 推荐执行顺序（最短路径）

1. 改 `merge_three_molds.json` 并执行合并
2. 改 `stage1_multimold_cotraining.json` 并跑阶段 1
3. 分别改 A/B/C 的 `dataset.episodes` 与 `pretrained_path`
4. 跑三次阶段 2

以上流程即可完成“先融合、后专属微调”的两阶段训练。
