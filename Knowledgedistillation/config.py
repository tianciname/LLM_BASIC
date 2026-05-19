# config.py
import os

# 模型路径或名称
TEACHER_MODEL_NAME = "Qwen/Qwen3.5-4B"
STUDENT_MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"

# 训练超参数
OUTPUT_DIR = "./student_model"
SEED = 42

# 蒸馏超参数
ALPHA = 0.5           # KL散度损失权重
TEMPERATURE = 2.0     # 温度系数

# 训练参数 (针对 RTX 5090 32G 显存优化)
EPOCHS = 15
BATCH_SIZE = 2                 # 每 GPU 的 micro batch size
GRADIENT_ACCUMULATION_STEPS = 8  # 有效 batch_size = 2*8 = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 512           # 512 够用且省显存，可调大到 768
FP16 = False
BF16 = True                    # 5090 完美支持 BF16
SAVE_STEPS = 500
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 2

# 数据集路径
TRAIN_DATA_PATH = "./dataset/train.jsonl"
EVAL_DATA_PATH = "./dataset/eval.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
