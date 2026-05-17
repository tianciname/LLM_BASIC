# Qwen-LLM 白盒知识蒸馏项目 (Knowledge Distillation)

本项目实现了一个高效的大语言模型白盒蒸馏流程。基于 KL 散度与交叉熵损失（Cross-Entropy Loss），将大参数量教师模型（Teacher Model）的能力蒸馏给小参数量学生模型（Student Model）。

本项目代码已深度适配 **NVIDIA RTX 5090 (32GB VRAM)** 级别的高端显卡，通过全量 GPU 计算与 Flash Attention 2 榨干显卡性能。

## 💡 核心特性
- **双模型 GPU 常驻**: 抛弃低效的 CPU 教师推理方案，利用 32G 显存将 Teacher 和 Student 均置于 GPU。
- **极致提速**: 全面引入 `bfloat16` (BF16) 精度、`Flash Attention 2` 与 `Fused AdamW`。
- **纯净日志**: 移除了第三方冗余日志插件，采用标准 Logging 结合 Tensorboard，终端输出干净明了。
- **灵活适配**: 默认使用 Qwen2.5 系列 (3B 蒸馏至 0.5B)，也可根据 `config.py` 轻松切换为 Llama、GLM 等任意 HuggingFace 架构模型。

## ⚙️ 环境安装

建议使用 Python 3.10+，执行以下命令安装依赖：

```bash
pip install -r requirements.txt