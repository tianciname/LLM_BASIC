import torch

# 配置文件
CONFIG = {
    # 被评估的 HuggingFace 模型名称或本地路径
    # 建议测试时使用小模型，如 "Qwen/Qwen2.5-0.5B" 或 "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "model_id": "Qwen/Qwen3.5-0.8B", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_samples": 100, # 每个基准测试的样本数，控制评估规模
    "max_new_token_length": 4096,
    "seed": 42
}