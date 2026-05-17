# prepare_dataset.py
import json
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["HF_TOKEN"] = "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"
# -------------------- 配置 --------------------
# 在这里选择你想要使用的数据集
USE_DATASETS = {
    'coig_cqia': False,      # 中文通用指令
    'magpie_qwen': True,    # 大规模通用对话（设为False暂不使用，可自行开启）
    'gsm8k': False,           # 英文数学推理 (CoT)
}
OUTPUT_TRAIN = './dataset/train.jsonl'
OUTPUT_EVAL = './dataset/eval.jsonl'
TRAIN_RATIO = 0.95  # 95% 数据作为训练集，5% 作为评估集

def write_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
# -------------------------------------------------

# 1. 处理 COIG-CQIA
if USE_DATASETS.get('coig_cqia'):
    print("Processing COIG-CQIA...")
    # load_dataset 会从 Hugging Face 下载
    coig_ds = load_dataset("m-a-p/COIG-CQIA", split="train") 
    coig_data = []
    for item in coig_ds:
        # 根据COIG-CQIA的实际字段名调整，这里示例提取 "instruction", "output"
        # 如果字段名不一致，你需要打印一条数据查看结构
        # print(coig_ds[0])
        if 'instruction' in item and 'output' in item:
            coig_data.append({
                "instruction": item['instruction'],
                "input": item.get('input', ''),
                "output": item['output']
            })
    print(f"  Loaded {len(coig_data)} samples from COIG-CQIA")
    # 随机打乱数据（使用hash保证每次运行打乱一致）
    coig_data = sorted(coig_data, key=lambda x: hash(x['instruction']))
    split_idx = int(len(coig_data) * TRAIN_RATIO)
    train_data_coig, eval_data_coig = coig_data[:split_idx], coig_data[split_idx:]

# 2. 处理 Magpie-Qwen2.5 (示例，同样需要先检查数据格式)
if USE_DATASETS.get('magpie_qwen'):
    print("Processing Magpie-Qwen2.5...")
    # 这是一个超大数据集，第一次加载可能需要较长时间
    magpie_ds = load_dataset("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1", split="train", streaming=True)
    magpie_data = []
    # 为方便演示，只取前20000条数据
    for i, item in enumerate(magpie_ds):
        if i >= 20000:
            break
        # Magpie数据集的格式通常是 {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        # 需要转换为 instruction/output
        if 'conversations' in item:
            try:
                user_content = None
                assistant_content = None
                for turn in item['conversations']:
                    if turn['from'] == 'human':
                        user_content = turn['value']
                    elif turn['from'] == 'gpt':
                        assistant_content = turn['value']
                if user_content and assistant_content:
                    magpie_data.append({
                        "instruction": user_content,
                        "input": "",
                        "output": assistant_content
                    })
            except:
                continue
    print(f"  Loaded {len(magpie_data)} samples from Magpie-Qwen2.5")
    # 打乱并划分
    magpie_data = sorted(magpie_data, key=lambda x: hash(x['instruction']))
    split_idx = int(len(magpie_data) * TRAIN_RATIO)
    train_data_magpie, eval_data_magpie = magpie_data[:split_idx], magpie_data[split_idx:]

# 3. 处理 GSM8K (数学CoT)
if USE_DATASETS.get('gsm8k'):
    print("Processing GSM8K...")
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_data = []
    # 对 GSM8K 数据进行 CoT 格式化，让模型学习思考过程
    for item in gsm8k_ds:
        # GSM8K 数据格式: {"question": "...", "answer": "...answer is \boxed{value}"}
        # 将答案解析为推理过程和最终答案
        answer_text = item['answer']
        # 简单的分离，实际中可能需要更复杂的处理
        parts = answer_text.split('####')
        if len(parts) == 2:
            reasoning, final_answer = parts[0].strip(), parts[1].strip()
            output_formatted = f"<think>\n{reasoning}\n</think>\n<response>\n{final_answer}\n</response>"
        else:
            output_formatted = answer_text # fallback
        
        gsm8k_data.append({
            "instruction": item['question'],
            "input": "",
            "output": output_formatted
        })
    print(f"  Loaded {len(gsm8k_data)} samples from GSM8K")
    # 打乱并划分
    gsm8k_data = sorted(gsm8k_data, key=lambda x: hash(x['instruction']))
    split_idx = int(len(gsm8k_data) * TRAIN_RATIO)
    train_data_gsm8k, eval_data_gsm8k = gsm8k_data[:split_idx], gsm8k_data[split_idx:]

# 4. 合并所有选中的数据集
final_train_data = []
final_eval_data = []

if USE_DATASETS.get('coig_cqia'):
    final_train_data.extend(train_data_coig)
    final_eval_data.extend(eval_data_coig)
if USE_DATASETS.get('magpie_qwen'):
    final_train_data.extend(train_data_magpie)
    final_eval_data.extend(eval_data_magpie)
if USE_DATASETS.get('gsm8k'):
    final_train_data.extend(train_data_gsm8k)
    final_eval_data.extend(eval_data_gsm8k)

# 5. 保存最终用于训练的数据文件
print(f"Final: {len(final_train_data)} training samples, {len(final_eval_data)} evaluation samples")
write_jsonl(final_train_data, OUTPUT_TRAIN)
write_jsonl(final_eval_data, OUTPUT_EVAL)
print("Data preparation done!")