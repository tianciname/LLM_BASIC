# LLM Multi-Dimensional Evaluator

此项目提供了一个开箱即用的自动化框架，用于在五个核心维度（知识、理解、代码、真实性、安全性）评估 HuggingFace 生态系统中的大型语言模型。

## 目录结构
```text
llm_eval_project/
├── main.py                # 主评估逻辑与模型加载
├── config.py              # 配置（模型ID、样本数量、硬件加速）
├── evaluators/            # 评估器模块
│   ├── mmlu_eval.py       # 知识推理
│   ├── hellaswag_eval.py  # 语言理解
│   ├── humaneval_eval.py  # 代码生成
│   ├── truthfulqa_eval.py # 事实真实性
│   └── safety_eval.py     # 安全判断
├── utils.py               # 雷达图渲染与Markdown生成
├── requirements.txt       # 环境依赖
└── README.md              # 项目文档