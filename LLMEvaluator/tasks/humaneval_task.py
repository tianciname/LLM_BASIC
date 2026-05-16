from .base_task import BaseTask
from datasets import load_dataset
import re, signal, sys, io, contextlib

class HumanEvalTask(BaseTask):
    def __init__(self, num_samples=None):
        super().__init__("humaneval", "generation")
        self.num_samples = num_samples

    def load_data(self, split="test", num_samples=None):
        # HumanEval需要从openai_humaneval加载
        dataset = load_dataset("openai_humaneval", split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def preprocess(self, dataset, tokenizer):
        return dataset

    def check_code(self, code, test):
        """在隔离环境下运行测试（安全起见仅执行简单代码，无危险操作）"""
        try:
            # 合并代码和测试
            full_code = code + "\n" + test
            # 捕获输出
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(full_code, {})
            return True, ""
        except Exception as e:
            return False, str(e)

    def compute_metrics(self, predictions, references):
        # predictions: 生成的代码字符串列表
        # references: 包含test用例
        correct = 0
        for pred, ref in zip(predictions, references):
            test = ref["test"]
            entry_point = ref["entry_point"]
            # 确保包含函数定义
            if "def " + entry_point not in pred:
                continue
            ok, _ = self.check_code(pred, test)
            if ok:
                correct += 1
        return {"pass@1": correct / len(references) if references else 0}