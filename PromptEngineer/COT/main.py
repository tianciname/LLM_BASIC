import random
import torch
import os
import sys

import config
from tasks import get_task
from data_loader import load_few_shot_examples
from model import load_model_and_tokenizer
from experiments.experiment1_cot_comparison import Experiment1
from experiments.experiment2_step_impact import Experiment2
from experiments.experiment3_shot_number import Experiment3


def main():
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    model, tokenizer, model_type = load_model_and_tokenizer()
    few_shot_pool = load_few_shot_examples()

    task_names = ["arithmetic", "commonsense"]
    tasks = [get_task(name) for name in task_names]

    if config.RUN_EXPERIMENT_1:
        exp1 = Experiment1()
        exp1.run(tasks, model, tokenizer, model_type, few_shot_pool, config)

    if config.RUN_EXPERIMENT_2:
        exp2 = Experiment2()
        exp2.run(tasks, model, tokenizer, model_type, few_shot_pool, config)

    if config.RUN_EXPERIMENT_3:
        exp3 = Experiment3()
        exp3.run(tasks, model, tokenizer, model_type, few_shot_pool, config)

    generate_plots()


def generate_plots():
    import json
    output_dir = "output"

    # 实验1 图表
    exp1_dir = os.path.join(output_dir, "experiment1")
    if os.path.exists(exp1_dir):
        exp1_results = {}
        for fname in os.listdir(exp1_dir):
            if fname.endswith("_random_results.json"):
                task_name = fname.replace("_random_results.json", "")
                with open(os.path.join(exp1_dir, fname), "r") as f:
                    exp1_results[task_name] = json.load(f)
        if exp1_results:
            from visualization import plot_exp1
            plot_exp1(exp1_results, os.path.join(exp1_dir, "exp1_comparison.png"))

    # 实验2 图表
    exp2_dir = os.path.join(output_dir, "experiment2")
    if os.path.exists(exp2_dir):
        exp2_results = {}
        for fname in os.listdir(exp2_dir):
            if fname.endswith("_step_impact_results.json"):
                task_name = fname.replace("_step_impact_results.json", "")
                with open(os.path.join(exp2_dir, fname), "r") as f:
                    exp2_results[task_name] = json.load(f)
        if exp2_results:
            from visualization import plot_exp2
            plot_exp2(exp2_results, os.path.join(exp2_dir, "exp2_step_impact.png"))

    # 实验3 图表
    exp3_dir = os.path.join(output_dir, "experiment3")
    if os.path.exists(exp3_dir):
        exp3_results = {}
        for fname in os.listdir(exp3_dir):
            if fname.endswith("_shot_count_results.json"):
                task_name = fname.replace("_shot_count_results.json", "")
                with open(os.path.join(exp3_dir, fname), "r") as f:
                    exp3_results[task_name] = json.load(f)
        if exp3_results:
            from visualization import plot_exp3
            plot_exp3(exp3_results, os.path.join(exp3_dir, "exp3_shot_impact.png"))


if __name__ == "__main__":
    main()
