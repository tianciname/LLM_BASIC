import os
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    def __init__(self, name, output_dir="output"):
        self.name = name
        self.output_dir = os.path.join(output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def run(self, tasks, model, tokenizer, model_type, few_shot_pool, config):
        pass