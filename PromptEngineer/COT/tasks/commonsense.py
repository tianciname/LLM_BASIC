from .base_task import BaseTask
from data_loader import load_boolq_samples


class CommonsenseTask(BaseTask):
    def __init__(self):
        super().__init__("Commonsense")

    def load_samples(self, sample_size, random_seed, steps=None):
        return load_boolq_samples(sample_size, random_seed, steps=steps)