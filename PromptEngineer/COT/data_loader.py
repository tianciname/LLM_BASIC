import json
from pathlib import Path
from datasets import load_dataset
import random
import re
import sys

DATA_DIR = Path(__file__).parent / "data"

def load_few_shot_examples():
    file_path = DATA_DIR / "few_shot_examples.json"
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples

def load_gsm8k_samples(size, seed, steps=None):
    """
    加载 GSM8K 样本，若指定 steps 则只抽取符合步数的样本。
    若符合条件的样本不足，返回实际能抽到的数量（最多 size），并打印警告。
    """
    random.seed(seed)
    dataset = load_dataset("gsm8k", "main", split="train")
    all_samples = []
    for item in dataset:
        q = item["question"]
        ans_raw = item["answer"]
        final = re.search(r'####\s*([\d\.\-]+)', ans_raw)
        ans = final.group(1) if final else ans_raw.strip()
        lines = [l for l in ans_raw.split('\n') if l.strip() and '####' not in l]
        s = max(1, len(lines) // 2)
        all_samples.append({"question": q, "answer": ans, "steps": s})

    if steps is not None:
        if isinstance(steps, int):
            steps = [steps]
        filtered = [s for s in all_samples if s["steps"] in steps]
        available = len(filtered)
        if available < size:
            print(f"[警告] GSM8K 中步数为 {steps} 的样本只有 {available} 个，请求 {size} 个，将使用全部 {available} 个。", file=sys.stderr)
            size = available
        return random.sample(filtered, size)
    return random.sample(all_samples, size)

def load_boolq_samples(size, seed, steps=None):
    """
    加载 BoolQ 样本，若指定 steps 则只抽取符合步数的样本。
    若符合条件的样本不足，返回实际能抽到的数量（最多 size），并打印警告。
    """
    random.seed(seed)
    dataset = load_dataset("boolq", split="train")
    all_samples = []
    for idx, item in enumerate(dataset):
        q = item["question"]
        ans = "yes" if item["answer"] else "no"
        s = 1 if idx % 2 == 0 else 2
        all_samples.append({"question": q, "answer": ans, "steps": s})

    if steps is not None:
        if isinstance(steps, int):
            steps = [steps]
        filtered = [s for s in all_samples if s["steps"] in steps]
        available = len(filtered)
        if available < size:
            print(f"[警告] BoolQ 中步数为 {steps} 的样本只有 {available} 个，请求 {size} 个，将使用全部 {available} 个。", file=sys.stderr)
            size = available
        return random.sample(filtered, size)
    return random.sample(all_samples, size)

def load_uniform_steps_samples(load_fn, total_size, steps_list, seed):
    """
    从数据集中按步数均匀抽样，总样本数尽量接近 total_size，
    各个步数尽量平均分配。若某步数样本不足，则返回实际数量，其他步数不变。
    """
    random.seed(seed)
    n_steps = len(steps_list)
    base = total_size // n_steps
    remainder = total_size % n_steps
    samples = []
    for i, step in enumerate(steps_list):
        num = base + 1 if i < remainder else base
        # load_fn 在内部处理不足的情况，返回实际数量
        batch = load_fn(num, seed=None, steps=step)  # 注意 seed 已在函数外部设定，内部使用当前随机状态
        if batch:
            samples.extend(batch)
    random.shuffle(samples)
    return samples