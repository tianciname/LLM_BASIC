import os, json
from .base_experiment import BaseExperiment
from prompts import build_direct_prompt, build_zero_cot_prompt, build_structured_cot_prompt
from evaluation import evaluate_single

class Experiment1(BaseExperiment):
    def __init__(self):
        super().__init__("experiment1")

    def run(self, tasks, model, tokenizer, model_type, few_shot_pool, config):
        for task in tasks:
            required = config.MATH_SAMPLE_SIZE if task.name == "Arithmetic" else config.COMMONSENSE_SAMPLE_SIZE
            # 直接随机抽样，不按步数过滤
            samples = task.load_samples(required, config.RANDOM_SEED)

            log_path = os.path.join(self.output_dir, f"{task.name}_random.txt")
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"实验1: 随机抽样（所有步数）下 CoT 类型对比 - 任务: {task.name}\n")
                baselines = [
                    ("Direct", build_direct_prompt),
                    ("Zero-shot CoT", build_zero_cot_prompt),
                    ("Structured CoT", build_structured_cot_prompt)
                ]
                results = {}
                for name, builder in baselines:
                    acc = evaluate_single(samples, builder, model, tokenizer, model_type, log, prefix=name)
                    results[name] = acc
                log.write(f"基线结果: {results}\n")

            # 保存 JSON
            with open(os.path.join(self.output_dir, f"{task.name}_random_results.json"), "w") as f:
                json.dump(results, f, indent=2)

            print(f"[实验1] {task.name}: 随机抽样样本数 = {len(samples)}")