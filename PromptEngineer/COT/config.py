# 模型与生成参数配置
MODEL_NAME = "Qwen/Qwen1.5-0.5B"              
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1
DO_SAMPLE = False
RANDOM_SEED = 42
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"


# 任务样本数量（每个任务从数据集中抽样数）
MATH_SAMPLE_SIZE = 50
COMMONSENSE_SAMPLE_SIZE = 50

# 实验选择
RUN_EXPERIMENT_1 = True   # CoT类型对比
RUN_EXPERIMENT_2 = True   # 不同步数影响
RUN_EXPERIMENT_3 = True   # 提示词数量影响（步数×shot数）


# 实验2配置：少样本示例步数列表（如 [1,2,3]），将分别评估仅含这些步数的示例
EXP2_STEPS = [1,2,3,4]   # 每个步数都会生成一种少样本配置

# 实验3配置：步数列表 × 提示词数量列表
EXP3_STEPS = [4]
EXP3_SHOT_COUNTS = [2, 4, 6, 8, 10]    # 三种提示词数量
# 实验3将在每种步数下，变化shot数量，绘制折线图。


# 可视化输出格式
PLOT_FORMAT = 'png'