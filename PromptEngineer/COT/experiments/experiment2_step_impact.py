import os, random, json
from .base_experiment import BaseExperiment
from prompts import build_few_shot_prompt
from evaluation import evaluate_single

class Experiment2(BaseExperiment):
    def __init__(self):
        super().__init__("experiment2")

    def run(self, tasks, model, tokenizer, model_type, few_shot_pool, config):
        steps_list = config.EXP2_STEPS
        for task in tasks:
            sample_size = config.MATH_SAMPLE_SIZE if task.name == "Arithmetic" else config.COMMONSENSE_SAMPLE_SIZE
            samples = task.load_samples(sample_size, config.RANDOM_SEED)

            # 从实验1读取基线（若存在）
            baseline_path = os.path.join("output", "experiment1", f"{task.name}_random_results.json")
            baselines = {}
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baselines = json.load(f)
            else:
                print(f"警告: 未找到实验1基线文件 {baseline_path}，实验2将不绘制基线。")

            log_path = os.path.join(self.output_dir, f"{task.name}_step_impact.txt")
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"实验2: 示例步数影响 - 任务: {task.name}\n")
                few_shot_acc = {}
                for step in steps_list:
                    # 筛选指定步数的示例
                    candidates = [ex for ex in few_shot_pool if ex.get("steps", 1) == step]
                    if len(candidates) < 4:
                        others = [ex for ex in few_shot_pool if ex.get("steps", 1) != step]
                        candidates += others[:4 - len(candidates)]
                    random.shuffle(candidates)
                    examples = candidates[:4]
                    builder = lambda q, exs=examples: build_few_shot_prompt(q, exs, 4)
                    acc = evaluate_single(samples, builder, model, tokenizer, model_type, log, prefix=f"Step-{step}-only 4-shot")
                    few_shot_acc[step] = acc
                    log.write(f"步数 {step} 的 4-shot 准确率: {acc:.2%}\n")

            result = {"baselines": baselines, "few_shot_by_step": few_shot_acc}
            with open(os.path.join(self.output_dir, f"{task.name}_step_impact_results.json"), "w") as f:
                json.dump(result, f, indent=2)

            print(f"[实验2] {task.name}: 完成，结果已保存")