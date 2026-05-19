import os, random, json
from .base_experiment import BaseExperiment
from prompts import build_few_shot_prompt
from evaluation import evaluate_single

class Experiment3(BaseExperiment):
    def __init__(self):
        super().__init__("experiment3")

    def run(self, tasks, model, tokenizer, model_type, few_shot_pool, config):
        steps = config.EXP3_STEPS
        shot_counts = config.EXP3_SHOT_COUNTS
        for task in tasks:
            sample_size = config.MATH_SAMPLE_SIZE if task.name == "Arithmetic" else config.COMMONSENSE_SAMPLE_SIZE
            samples = task.load_samples(sample_size, config.RANDOM_SEED)

            # 从实验1读取基线
            baseline_path = os.path.join("output", "experiment1", f"{task.name}_random_results.json")
            baselines = {}
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baselines = json.load(f)
            else:
                print(f"警告: 未找到实验1基线文件 {baseline_path}，实验3将不绘制基线。")

            log_path = os.path.join(self.output_dir, f"{task.name}_shot_count.txt")
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"实验3: 提示词数量影响 - 任务: {task.name}\n")
                result_by_step = {}
                for step in steps:
                    candidates = [ex for ex in few_shot_pool if ex.get("steps", 1) == step]
                    if len(candidates) < max(shot_counts):
                        others = [ex for ex in few_shot_pool if ex.get("steps", 1) != step]
                        candidates += others[:max(shot_counts) - len(candidates)]
                    random.shuffle(candidates)
                    step_accs = {}
                    for k in shot_counts:
                        exs = candidates[:k]
                        builder = lambda q, exs=exs, k=k: build_few_shot_prompt(q, exs, k)
                        acc = evaluate_single(samples, builder, model, tokenizer, model_type, log, prefix=f"Step{step}-{k}shot")
                        step_accs[k] = acc
                        log.write(f"步数 {step} {k}-shot 准确率: {acc:.2%}\n")
                    result_by_step[step] = step_accs

            final = {"baselines": baselines, "by_step": result_by_step}
            with open(os.path.join(self.output_dir, f"{task.name}_shot_count_results.json"), "w") as f:
                json.dump(final, f, indent=2)

            print(f"[实验3] {task.name}: 完成，结果已保存")