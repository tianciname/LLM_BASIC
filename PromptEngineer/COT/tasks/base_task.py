from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def load_samples(self, sample_size, random_seed):
        """返回样本列表，每个样本包含 question, answer, steps"""
        pass