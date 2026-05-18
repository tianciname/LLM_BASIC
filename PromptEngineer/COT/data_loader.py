import json
from pathlib import Path
from datasets import load_dataset
import random
import re
import sys

DATA_DIR = Path(__file__).parent / "data"

# ----------------- 加载少样本示例池（不变） -----------------
def load_few_shot_examples():
    file_path = DATA_DIR / "few_shot_examples.json"
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples

# ----------------- 按步数过滤测试样本（不变） -----------------
def filter_samples_by_steps(samples, steps):
    if steps is None or (isinstance(steps, list) and len(steps) == 0):
        return samples
    if isinstance(steps, int):
        steps = [steps]
    return [s for s in samples if s.get("steps", 1) in steps]

# ----------------- 内置备用测试样本（独立定义） -----------------
def _get_builtin_math_samples():
    """算术备用样本（GSM8K 加载失败时使用）"""
    return [
        {"question": "一辆车以每小时60英里的速度行驶，2.5小时后能走多远？", "answer": "150", "steps": 1},
        {"question": "约翰有45颗弹珠，他在游戏中输了15颗，然后朋友给了他20颗，他现在有多少颗？", "answer": "50", "steps": 2},
        {"question": "一张披萨切成8等份，吃掉3块，剩下的占整张披萨的几分之几？", "answer": "5/8", "steps": 1},
        {"question": "一袋糖果有120颗，平均分给10个孩子，每个孩子得到多少颗？", "answer": "12", "steps": 1},
        {"question": "火车下午2:30出发，下午5:45到达，行程共多少分钟？", "answer": "195", "steps": 2},
    ]

def _get_builtin_commonsense_samples():
    """常识备用样本（BoolQ 加载失败时使用）"""
    return [
        {"question": "香蕉是水果吗？", "answer": "是", "steps": 1},
        {"question": "能在北极找到企鹅吗？", "answer": "不能", "steps": 2},
        {"question": "珠穆朗玛峰是地球上最高的山吗？", "answer": "是", "steps": 1},
        {"question": "太阳从西边升起吗？", "answer": "不是", "steps": 1},
        {"question": "猫会像狗一样汪汪叫吗？", "answer": "不会", "steps": 2},
    ]

# ----------------- 真实数据集加载（独立容错） -----------------
def load_real_dataset_samples(math_size=50, commonsense_size=50, random_seed=42):
    """
    分别尝试加载 GSM8K 和 BoolQ，若某个加载失败则用内置备用数据替换。
    返回：[("算术推理", 样本列表), ("常识推理", 样本列表)]
    """
    random.seed(random_seed)

    # ---------- 加载 GSM8K ----------
    math_tasks = []
    try:
        print("正在加载 GSM8K 数据集...")
        gsm8k = load_dataset("gsm8k", "main", split="train")
        gsm8k_list = list(gsm8k)
        gsm8k_samples = random.sample(gsm8k_list, min(math_size, len(gsm8k_list)))
        for item in gsm8k_samples:
            question = item["question"]
            answer_raw = item["answer"]
            # 提取 #### 后的数字答案
            final_answer_match = re.search(r'####\s*([\d\.\-]+)', answer_raw)
            final_answer = final_answer_match.group(1) if final_answer_match else answer_raw.strip()
            # 粗略估算步数
            reasoning_lines = [line.strip() for line in answer_raw.split('\n') if line.strip() and '####' not in line]
            steps = max(1, len(reasoning_lines) // 2)
            math_tasks.append({"question": question, "answer": final_answer, "steps": steps})
        print(f"GSM8K 加载成功，共 {len(math_tasks)} 个样本")
    except Exception as e:
        print(f"GSM8K 加载失败: {e}，将使用内置备用算术样本。", file=sys.stderr)
        math_tasks = _get_builtin_math_samples()

    # ---------- 加载 BoolQ（替代 StrategyQA） ----------
    commonsense_tasks = []
    try:
        print("正在加载 BoolQ 数据集...")
        # BoolQ 数据集 ID 为 "boolq"，包含 train 和 validation
        boolq = load_dataset("boolq", split="train")
        boolq_list = list(boolq)
        boolq_samples = random.sample(boolq_list, min(commonsense_size, len(boolq_list)))
        for idx, item in enumerate(boolq_samples):
            question = item["question"]
            answer_bool = item["answer"]  # True / False
            answer_cn = "yes" if answer_bool else "no"   # 改为英文
            steps = 1 if idx % 2 == 0 else 2
            commonsense_tasks.append({"question": question, "answer": answer_cn, "steps": steps})
        print(f"BoolQ 加载成功，共 {len(commonsense_tasks)} 个样本")
    except Exception as e:
        print(f"BoolQ 加载失败: {e}，将使用内置备用常识样本。", file=sys.stderr)
        commonsense_tasks = _get_builtin_commonsense_samples()

    return [
        ("算术推理", math_tasks),
        ("常识推理", commonsense_tasks)
    ]