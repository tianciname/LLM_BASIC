from .base_task import BaseTask
from datasets import load_dataset
import re

class GSM8KTask(BaseTask):
    def __init__(self, num_samples=None):
        super().__init__("gsm8k", "generation")
        self.num_samples = num_samples

    def load_data(self, split="test", num_samples=None):
        dataset = load_dataset("gsm8k", "main", split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def preprocess(self, dataset, tokenizer):
        return dataset

    def extract_answer(self, text):
        # 提取最后出现的数字作为答案
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if matches:
            return float(matches[-1].replace(",", ""))
        return None

    def compute_metrics(self, predictions, references):
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_num = self.extract_answer(pred)
            # GSM8K答案格式: "#### 数字"
            ref_num = None
            ref_match = re.search(r"####\s*(-?\d+[.,]?\d*)", ref)
            if ref_match:
                ref_num = float(ref_match.group(1).replace(",", ""))
            if pred_num is not None and ref_num is not None and abs(pred_num - ref_num) < 1e-5:
                correct += 1
        return {"accuracy": correct / len(references) if references else 0}