# config.py
import os

# 模型路径或名称
TEACHER_MODEL_NAME = "/root/code/LLM_BASIC/LoRA-SFT/models/Qwen3.5-4B-Base"
STUDENT_MODEL_NAME = "Qwen/Qwen3.5-2B"

# 训练超参数
OUTPUT_DIR = "./student_model"
LOG_DIR = "./logs"
SEED = 42

# 蒸馏超参数
ALPHA = 0.7           # KL散度损失权重
TEMPERATURE = 2.0     # 温度系数

# 训练参数
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 128
FP16 = False
BF16 = False
SAVE_STEPS = 500
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 2

# 数据集路径
TRAIN_DATA_PATH = "./dataset/train.jsonl"
EVAL_DATA_PATH = "./dataset/eval.jsonl"

# 其他
USE_DEEPSPEED = False   # 是否使用DeepSpeed ZeRO-2/3
DEEPSPEED_CONFIG = "ds_config.json"

# 创建必要目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)