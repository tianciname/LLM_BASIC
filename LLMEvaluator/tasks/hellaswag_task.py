from .base_task import BaseTask
from datasets import load_dataset
import torch
import numpy as np

class HellaSwagTask(BaseTask):
    def __init__(self, num_samples=None):
        super().__init__("hellaswag", "multiple_choice")
        self.num_samples = num_samples

    def load_data(self, split="validation", num_samples=None):
        dataset = load_dataset("hellaswag", split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def preprocess(self, dataset, tokenizer):
        # 保留原始文本，不进行tokenize，在评估时动态计算
        return dataset

    def compute_metrics(self, predictions, references):
        acc = np.mean(np.array(predictions) == np.array(references))
        return {"accuracy": acc}

    def evaluate_model(self, model, tokenizer, dataset, batch_size=4):
        """模型预测：选择最合理的结尾"""
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                for example in batch:
                    ctx = example["ctx"]
                    endings = example["endings"]
                    label = int(example["label"]) if "label" in example else 0

                    # 计算每个结尾的ppl
                    scores = []
                    for ending in endings:
                        full_text = ctx + " " + ending
                        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        scores.append(loss.item())  # 交叉熵越小越好
                    pred = np.argmin(scores)
                    predictions.append(pred)
                    labels.append(label)
        return predictions, labels