from abc import ABC, abstractmethod
from datasets import Dataset
from typing import Dict, Any, Optional

class BaseTask(ABC):
    """所有评估任务的基类"""
    def __init__(self, task_name: str, task_type: str):
        self.task_name = task_name
        self.task_type = task_type  # "classification", "generation", "multiple_choice"
        self.metrics = []

    @abstractmethod
    def load_data(self, split: str = "validation", num_samples: Optional[int] = None) -> Dataset:
        """加载数据集，返回 HuggingFace Dataset"""
        pass

    @abstractmethod
    def preprocess(self, dataset: Dataset, tokenizer, **kwargs) -> Dataset:
        """将原始数据转化为模型输入格式"""
        pass

    @abstractmethod
    def compute_metrics(self, predictions, references, **kwargs) -> Dict[str, float]:
        """计算评估指标"""
        pass

    def get_prompt(self, example: Dict) -> str:
        """对于生成式模型，构造输入prompt"""
        return example.get("text", "")