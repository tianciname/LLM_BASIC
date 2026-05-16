from .base_task import BaseTask
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

class MNLITask(BaseTask):
    def __init__(self, num_samples=None):
        super().__init__("mnli", "classification")
        self.num_samples = num_samples

    def load_data(self, split, num_samples=None):
        # 处理 MNLI 验证集的 split 名称问题
        if split == "validation":
            split = "validation_matched"
        dataset = load_dataset("glue", "mnli", split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def preprocess(self, dataset, tokenizer, max_length=256):
        def tokenize_fn(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.map(lambda x: {"labels": x["label"]})
        return dataset

    def compute_metrics(self, predictions, references):
        preds = np.argmax(predictions, axis=-1)
        return {"accuracy": accuracy_score(references, preds)}