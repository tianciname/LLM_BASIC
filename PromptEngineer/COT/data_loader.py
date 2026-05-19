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


def _estimate_boolq_steps(question: str) -> int:
    """
    根据问题复杂度估算推理步数。
    - 简单短问题（≤60 字符，无否定/从句）→ 1 步
    - 中等问题或含否定 → 2 步
    - 复杂长问题（≥100 字符，含从句或双重条件）→ 3 步
    """
    q_len = len(question)
    has_negation = bool(re.search(r'\bnot\b|\bnever\b|\bno\b', question, re.IGNORECASE))
    has_clause = bool(re.search(r'\bwhich\b|\bthat\b|\bwhereas\b|\balthough\b|;|—', question, re.IGNORECASE))
    has_multiple_conditions = question.count(',') >= 2

    if q_len >= 100 or (has_clause and has_multiple_conditions):
        return 3
    elif q_len >= 60 or has_negation:
        return 2
    else:
        return 1


def load_gsm8k_samples(size, seed, steps=None):
    """
    加载 GSM8K 样本，若指定 steps 则只抽取符合步数的样本。
    """
    random.seed(seed)
    dataset = load_dataset("gsm8k", "main", split="train")
    all_samples = []
    for item in dataset:
        q = item["question"]
        ans_raw = item["answer"]
        final = re.search(r'####\s*([\d\.\-]+)', ans_raw)
        ans = final.group(1) if final else ans_raw.strip()
        # 估算步数：按换行分隔的推理步骤数（排除空白行和答案行）
        lines = [l for l in ans_raw.split('\n') if l.strip() and '####' not in l]
        s = max(1, len(lines) // 2)
        all_samples.append({"question": q, "answer": ans, "steps": s})

    if steps is not None:
        if isinstance(steps, int):
            steps = [steps]
        filtered = [s for s in all_samples if s["steps"] in steps]
        available = len(filtered)
        if available < size:
            print(f"[警告] GSM8K 中步数为 {steps} 的样本只有 {available} 个，请求 {size} 个，将使用全部 {available} 个。",
                  file=sys.stderr)
            size = available
        return random.sample(filtered, min(size, available))
    return random.sample(all_samples, min(size, len(all_samples)))


def load_boolq_samples(size, seed, steps=None):
    """
    加载 BoolQ 样本，使用问题复杂度启发式估算推理步数。
    """
    random.seed(seed)
    dataset = load_dataset("boolq", split="train")
    all_samples = []
    for item in dataset:
        q = item["question"]
        ans = "yes" if item["answer"] else "no"
        s = _estimate_boolq_steps(q)
        all_samples.append({"question": q, "answer": ans, "steps": s})

    if steps is not None:
        if isinstance(steps, int):
            steps = [steps]
        filtered = [s for s in all_samples if s["steps"] in steps]
        available = len(filtered)
        if available < size:
            print(f"[警告] BoolQ 中步数为 {steps} 的样本只有 {available} 个，请求 {size} 个，将使用全部 {available} 个。",
                  file=sys.stderr)
            size = available
        return random.sample(filtered, min(size, available))
    return random.sample(all_samples, min(size, len(all_samples)))
