from .base_task import BaseTask
from data_loader import load_gsm8k_samples

class ArithmeticTask(BaseTask):
    def __init__(self):
        super().__init__("Arithmetic")

    def load_samples(self, sample_size, random_seed, steps=None):
        return load_gsm8k_samples(sample_size, random_seed, steps=steps)