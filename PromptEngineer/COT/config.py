import torch

# 模型与生成参数
MODEL_NAME = "Qwen/Qwen3.5-2B"
MAX_NEW_TOKENS = 1024
DO_SAMPLE = False
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 任务样本数量（每个任务从数据集中抽样数）
MATH_SAMPLE_SIZE = 100
COMMONSENSE_SAMPLE_SIZE = 100

# 实验开关
RUN_EXPERIMENT_1 = True   # CoT 类型对比 (Direct / Zero-shot CoT / Structured CoT)
RUN_EXPERIMENT_2 = True   # 示例推理步数影响
RUN_EXPERIMENT_3 = True   # 提示词数量 × 步数交叉

# 实验2：评估的各步数配置
EXP2_STEPS = [1, 2, 3, 4]

# 实验3：步数 × shot 数量交叉
EXP3_STEPS = [4]
EXP3_SHOT_COUNTS = [2, 4, 6, 8, 10]

# 可视化
PLOT_FORMAT = 'png'
