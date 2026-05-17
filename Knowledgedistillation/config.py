# config.py
import os

# 模型路径或名称
TEACHER_MODEL_NAME = "Qwen/Qwen3.5-2B" # 建议用更新的 2.5 系列
STUDENT_MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"

# 训练超参数
OUTPUT_DIR = "./student_model"
SEED = 42

# 蒸馏超参数
ALPHA = 0.5           # KL散度损失权重 (通常0.5或0.3更平衡)
TEMPERATURE = 2.0     # 温度系数

# 训练参数 (针对 RTX 5090 32G 显存优化)
EPOCHS = 3
BATCH_SIZE = 1   # 5090 可以轻松跑 BATCH_SIZE=4 甚至 8 (取决于序列长度)
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 1024 # 如果需要长文本，5090可提升至 2048 或 4096
FP16 = False
BF16 = True           # 5090 完美支持 BF16，防溢出且速度快
SAVE_STEPS = 500
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 2

# 数据集路径
TRAIN_DATA_PATH = "./dataset/train.jsonl"
EVAL_DATA_PATH = "./dataset/eval.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)