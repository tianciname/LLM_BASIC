# 思维链（CoT）提示策略对比与少样本敏感性分析

## 1. 实验目的

在本地使用 Hugging Face + PyTorch 加载语言模型，系统对比不同思维链（Chain-of-Thought）提示策略在多个推理任务上的表现，并重点分析模型对少样本示例的**数量**、**内容**以及**推理步数**的敏感性。

## 2. 实验环境与模型

- 框架：`transformers`、`torch`、`datasets`
- 模型：可配置的因果语言模型（默认 `Qwen/Qwen1.5-0.5B`）
- 生成设置：`do_sample = False`（贪婪解码），保证结果可复现
- 设备：自动选择 GPU（可用时）或 CPU

## 3. 项目结构

```
COT/
├── main.py                  # 入口：调度实验 + 生成可视化
├── config.py                # 所有可配置参数
├── model.py                 # 模型加载与推理
├── prompts.py               # 4 种提示词构建策略（格式统一）
├── evaluation.py            # 评估逻辑 + 答案提取 + 结果缓存
├── data_loader.py           # 数据加载（GSM8K / BoolQ，含步数启发式估算）
├── visualization.py         # matplotlib 折线图（双任务合并，统一配色）
├── data/
│   └── few_shot_examples.json   # 50 个英文手工标注示例池（1~5 步）
├── tasks/
│   ├── __init__.py          # 任务注册表
│   ├── base_task.py         # 抽象基类
│   ├── arithmetic.py        # 算术任务（GSM8K）
│   └── commonsense.py       # 常识任务（BoolQ）
├── experiments/
│   ├── base_experiment.py           # 实验基类
│   ├── experiment1_cot_comparison.py # 实验1：CoT 类型对比
│   ├── experiment2_step_impact.py    # 实验2：示例步数影响
│   └── experiment3_shot_number.py   # 实验3：Shot 数量 × 步数
├── output/                   # 结果输出（JSON / TXT / PNG）
└── .cache/                   # 模型输出缓存（可删除以强制重新推理）
```

## 4. 提示策略设计

共实现 **4 种提示策略**，格式统一使用 `Question:` / `Answer:` 前缀：

| 策略 | 模板 |
|------|------|
| **Direct** | `Question: {q}\nAnswer:` |
| **Zero-shot CoT** | `Question: {q}\nAnswer: Let's think step by step.` |
| **Structured CoT** | 强制要求输出推理过程，以 `Answer:` 后给出最终答案 |
| **Few-shot CoT** | 从 50 个示例池中选取 k 个 `Question:/Answer:` 对作为前缀 |

- 所有少样本示例来自内置的 **50 个英文人工标注样例**，覆盖 1～5 步推理，每步 10 个，涵盖算术、常识、逻辑等类型。
- Few-shot 示例通过池筛选 + 随机抽取确保多样性。

## 5. 任务与数据

| 任务 | 类型 | 数据来源 | 答案形式 | 默认抽样数 |
|------|------|----------|----------|-----------|
| **Arithmetic** | 数值计算 | GSM8K (train) | 数字 | `MATH_SAMPLE_SIZE=50` |
| **Commonsense** | 是非判断 | BoolQ (train) | `yes` / `no` | `COMMONSENSE_SAMPLE_SIZE=50` |

- GSM8K 步数通过解答文本的推理行数估算。
- BoolQ 步数通过问题复杂度启发式估算（长度、否定词、从句结构等）。

## 6. 评估方式

- **答案提取**：从模型输出中定位 `Answer:` / `answer is` 等标记，提取最终答案部分，避免全文子串匹配的干扰。
- **数字匹配**：数值归一化后做容差比较（< 0.001），自动忽略单位（yuan, meters, % 等）。
- **yes/no 匹配**：词边界正则匹配完整单词。
- **结果缓存**：模型输出按 `(策略, 样本)` 缓存至 `.cache/` 目录，重复运行自动跳过推理。
- **指标**：精确匹配准确率（Exact Match）。
- **日志**：每个实验生成独立 `.txt` 日志文件和 `.json` 结果文件。

## 7. 三个实验

### 实验1：CoT 类型对比
对比 Direct、Zero-shot CoT、Structured CoT 在随机抽样上的准确率。

### 实验2：示例步数影响
固定 4-shot，变化示例的推理步数（1/2/3/4-step），分析示例步数分布对模型推理的影响，以 Direct 为基线。

### 实验3：Shot 数量 × 步数交叉
固定示例步数（默认 step=4），变化 shot 数量（2/4/6/8/10），分析 shot scaling 效应。

## 8. 可视化

所有图表为**折线图**，两任务合并于同一图中，用不同颜色和线型区分：

- **Arithmetic**：暖色 coral (`#E8795A`)，实线 + 圆形标记
- **Commonsense**：冷色 steel blue (`#5B9BD5`)，虚线 + 方形标记
- **Direct 基线**：灰色虚线

生成的图表：
- `exp1_comparison.png` — CoT 策略对比
- `exp2_step_impact.png` — 示例步数影响
- `exp3_shot_impact.png` — Shot 数量影响

## 9. 可配置项

通过 `config.py` 文件调整：

```python
MODEL_NAME = "Qwen/Qwen1.5-0.5B"   # 模型名称
MATH_SAMPLE_SIZE = 50               # 算术任务样本数
COMMONSENSE_SAMPLE_SIZE = 50        # 常识任务样本数
RUN_EXPERIMENT_1 = True             # 实验1 开关
RUN_EXPERIMENT_2 = True             # 实验2 开关
RUN_EXPERIMENT_3 = True             # 实验3 开关
EXP2_STEPS = [1, 2, 3, 4]          # 实验2 步数列表
EXP3_STEPS = [4]                    # 实验3 步数
EXP3_SHOT_COUNTS = [2, 4, 6, 8, 10] # 实验3 shot 数量
```

## 10. 运行

```bash
pip install -r requirements.txt
python main.py
```

## 11. 结论要点

- CoT 提示（Zero-shot / Few-shot）相比 Direct 通常提升推理准确率。
- 示例数量增加可能带来提升，但存在饱和现象。
- 示例的推理步数分布显著影响模型表现：多步示例在复杂推理上更优，简单示例在简单题上更稳定。
- Structured CoT 强制分离推理与答案，输出更可控。
- 模型对示例选择敏感，相同 shot 数不同示例可导致性能波动。
