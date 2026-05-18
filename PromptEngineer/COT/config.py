# 模型与生成参数配置
MODEL_NAME = "Qwen/Qwen3.5-9B"                # 可替换为 "google/flan-t5-base" 等
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
DO_SAMPLE = False
RANDOM_SEED = 42
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# ========== 数据集采样配置 ==========
# 每个任务抽取的样本数量
MATH_SAMPLE_SIZE = 2           # GSM8K 样本数
COMMONSENSE_SAMPLE_SIZE = 2    # StrategyQA 样本数

# 混合示例的步数范围（用于随机抽取的候选池），设为 None 则使用全部示例
FEW_SHOT_STEP_RANGE = [3]   # 示例步数只从1~3步中选取，可根据需要修改

# 测试集步数过滤：None 表示全部，1 表示仅一步，2 表示两步，3 表示三步……
# 若要评估多步，可设为 [3,4,5] 等。留空列表或 None 表示全部。
EVAL_STEPS = [1, 2]       # 例如：None, 1, 2, [1,2], [3,4,5]

# 实验开关
RUN_MAIN_EXPERIMENT = True         # 主实验：全部策略评估 + 总体/步数分组结果
RUN_STEP_VERIFICATION = True      # 步数验证实验：不同示例步数 vs 测试步数

