# LLM Capability Evaluator

评估基于 HuggingFace Transformers 的大模型在多个能力维度的表现。

---

# 支持的任务

- **自然语言理解**（MNLI 三分类）
- **阅读理解**（SQuAD v2）
- **常识推理**（HellaSwag 选结尾）
- **数学推理**（GSM8K 算术题）
- **代码生成**（HumanEval pass@1）

---

# 安装

```bash
pip install -r requirements.txt
```

---

# 使用示例

## 基础评估

```bash
python run_eval.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks mnli hellaswag gsm8k \
    --num_samples 50 \
    --batch_size 4
```

## 量化模型评估

```bash
python run_eval.py \
    --model TheBloke/Llama-2-7B-GPTQ \
    --device cuda \
    --load_in_8bit
```

---

# 添加自定义任务

继承 `BaseTask`，实现以下方法：

- `load_data`
- `preprocess`
- `compute_metrics`

然后在 `run_eval.py` 中注册即可。

---

# 项目特点

## 多能力覆盖

支持：

- NLU
- 阅读理解
- 常识推理
- 数学推理
- 代码生成

## 模型结构自动适配

自动识别：

- causal language model
- seq2seq model
- classification model

## 可扩展架构

基于任务基类设计。

新增任务时，只需补全：

- 数据加载
- 预处理
- 指标计算

## 高效批处理

- 支持 `DataLoader`
- GPU 加速
- Batch 推理

## 结果持久化

自动输出 JSON 格式评估报告。

---

# 可扩展方向

该项目框架可以直接运行（需相应数据集已缓存或下载），并可按需求扩展更多任务，例如：

- 翻译（BLEU）
- 对话评估
- 多语言能力
- Agent 能力
- Tool Calling
- 长上下文测试

---

# 可视化工具

支持模型评估结果可视化。

---

# 1. 单模型雷达图

```bash
python visualize.py \
    --results results.json \
    --model_name "Llama-2-7b" \
    --type radar \
    --output llama2_radar.png
```

## 效果

一个五边形雷达图，各维度显示模型得分。

示例效果图：

https://i.imgur.com/6sJ9lWn.png

（实际图片由代码生成）

---

# 2. 多模型柱状图对比

```bash
python visualize.py \
    --type bar \
    --compare results_model_a.json results_model_b.json \
    --metric accuracy \
    --output model_comparison.png
```

功能：

- 自动读取多个 JSON
- 提取任务指标
- 绘制分组柱状图
- 对比不同模型表现

---

# 集成到主评估流程

在 `run_eval.py` 末尾自动调用可视化：

```python
from visualize import plot_radar, extract_radar_values

# 评估结束后
labels, values = extract_radar_values(evaluator.results)

plot_radar(
    labels,
    values,
    model_name,
    save_path="radar.png"
)
```

---

# 定制化

## 指标切换

修改：

```python
TASK_METRIC_MAP
```

即可切换主指标。

例如：

- SQuAD 使用 `exact_match`
- 分类任务使用 `accuracy`

---

## 样式美化

支持：

- seaborn 主题
- 自定义颜色
- 字体调整
- 图像尺寸调整

---

## 增加新任务

在以下配置中新增条目：

```python
TASK_METRIC_MAP
TASK_LABELS
```

即可完成扩展。

---

# 总结

这套评估框架具备：

- 多任务统一评估
- 模块化任务扩展
- 高效批量推理
- 自动结果统计
- 可视化分析

能够快速发现模型在：

- 语言理解
- 推理
- 数学
- 代码

等能力维度上的优势与短板。


## 指令
对比并可视化的完整指令如下（假设已安装项目依赖，模型名和路径按实际调整）：

### 1. 先分别评估两个模型，生成各自的结果 JSON

```bash
# 模型 A
python run_eval.py \
    --model /root/code/LLM_BASIC/LoRA-SFT/merged_model \
    --tasks mnli squad hellaswag gsm8k humaneval \
    --num_samples 100 \
    --batch_size 4 \
    --output results_sft-qwen3.5-4B.json

# 模型 B
python run_eval.py \
    --model /root/code/LLM_BASIC/LoRA-SFT/models/Qwen3.5-4B-Base\
    --tasks mnli squad hellaswag gsm8k humaneval \
    --num_samples 100 \
    --batch_size 4 \
    --output results_-qwen3.5-4B.json
```

### 2. 生成多模型对比柱状图（主要能力横向比较）

```bash
python visualize.py \
    --type bar \
    --compare results_sft-qwen3.5-4B.json results_-qwen3.5-4B.json \
    --metric accuracy \
    --output comparison_accuracy.png
```

若想看 SQuAD 的 F1 对比，可改 `--metric f1`。

### 3. （可选）单独生成每个模型的雷达图

```bash
python visualize.py --type radar --results results_sft-qwen3.5-4B.json --model_name "Llama-2-7b" --output llama2_radar.png
python visualize.py --type radar --results results_-qwen3.5-4B.json --model_name "Mistral-7B" --output mistral_radar.png
```

执行后会在当前目录下得到 `comparison_accuracy.png` 等多张图表，即可直观比较模型在各维度的能力差异。