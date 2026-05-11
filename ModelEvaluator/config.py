# config.py
from mmengine.config import read_base

# ================= 模型配置 =================
# 支持 HuggingFace 模型 ID 或本地路径
MODEL_PATH = "Qwen/Qwen2.5-0.5B"          # 你的模型路径或HuggingFace ID
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B"      # 一般与模型路径相同
MODEL_TYPE = "chat"                       # "chat" 或 "base"
MODEL_ABBR = "my_model"                   # 结果显示时的名称
MAX_OUT_LEN = 1024
MAX_SEQ_LEN = 4096
BATCH_SIZE = 4                            # 根据显存调整
NUM_GPUS = 1

# ================= 数据集配置 =================
# 可以从下方默认列表中选择，也可通过命令行传递
DEFAULT_DATASETS = ["mmlu", "ceval", "gsm8k"]   # 支持: mmlu, ceval, gsm8k, humaneval, hellaswag 等

# ================= 输出目录 =================
WORK_DIR = "./outputs/eval_results"

# ================= 模型构造 =================
def get_models():
    from opencompass.models import HuggingFacewithChatTemplate, HuggingFaceBaseModel
    if MODEL_TYPE == "chat":
        model = dict(
            type=HuggingFacewithChatTemplate,
            abbr=MODEL_ABBR,
            path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            max_out_len=MAX_OUT_LEN,
            max_seq_len=MAX_SEQ_LEN,
            batch_size=BATCH_SIZE,
            model_kwargs=dict(
                device_map='auto',
                trust_remote_code=True,
                dtype='auto',
            ),
            run_cfg=dict(num_gpus=NUM_GPUS),
        )
    else:  # base 模型
        model = dict(
            type=HuggingFaceBaseModel,
            abbr=MODEL_ABBR,
            path=MODEL_PATH,
            model_kwargs=dict(device_map='auto', trust_remote_code=True),
            tokenizer_kwargs=dict(trust_remote_code=True),
            max_out_len=MAX_OUT_LEN,
            max_seq_len=MAX_SEQ_LEN,
            batch_size=BATCH_SIZE,
            run_cfg=dict(num_gpus=NUM_GPUS),
        )
    return [model]

# ================= 数据集构造 =================
def get_datasets(dataset_names):
    datasets = []
    for name in dataset_names:
        if name == "mmlu":
            from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
            datasets += mmlu_datasets
        elif name == "ceval":
            from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
            datasets += ceval_datasets
        elif name == "gsm8k":
            from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
            datasets += gsm8k_datasets
        elif name == "humaneval":
            from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
            datasets += humaneval_datasets
        elif name == "hellaswag":
            from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
            datasets += hellaswag_datasets
        # 可继续添加...
        else:
            print(f"⚠️ 不支持的数据集: {name}，已跳过")
    # 统一设置生成长度
    for d in datasets:
        if 'infer_cfg' in d and 'inferencer' in d['infer_cfg']:
            d['infer_cfg']['inferencer']['max_out_len'] = MAX_OUT_LEN
    return datasets