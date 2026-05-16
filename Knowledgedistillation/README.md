# Qwen2.5 蒸馏项目：4B → 2B

本项目实现了将 **Qwen3.5-4B-Instruct** 蒸馏到 **Qwen3.5-2B-Instruct** 的完整流程。使用KL散度+交叉熵损失进行白盒蒸馏。

## 环境配置

```bash
pip install -r requirements.txt

