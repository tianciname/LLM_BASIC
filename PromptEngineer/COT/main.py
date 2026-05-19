import random, torch, os, sys
from config import *
from tasks import get_task
from data_loader import load_few_shot_examples
from model import load_model_and_tokenizer
from experiments.experiment1_cot_comparison import Experiment1
from experiments.experiment2_step_impact import Experiment2
from experiments.experiment3_shot_number import Experiment3
import json
os.environ["HF_TOKEN"] = "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"
def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 加载模型
    model, tokenizer, model_type = load_model_and_tokenizer()
    # 加载少样本池
    few_shot_pool = load_few_shot_examples()

    # 准备任务列表（从配置中读取要评估的任务，这里默认全部）
    task_names = ["arithmetic", "commonsense"]   # 可改为从配置读取
    tasks = [get_task(name) for name in task_names]

    # 实验1
    if RUN_EXPERIMENT_1:
        exp1 = Experiment1()
        exp1.run(tasks, model, tokenizer, model_type, few_shot_pool, config=sys.modules[__name__])

    # 实验2
    if RUN_EXPERIMENT_2:
        exp2 = Experiment2()
        exp2.run(tasks, model, tokenizer, model_type, few_shot_pool, config=sys.modules[__name__])

    # 实验3
    if RUN_EXPERIMENT_3:
        exp3 = Experiment3()
        exp3.run(tasks, model, tokenizer, model_type, few_shot_pool, config=sys.modules[__name__])

    # 后处理：生成可视化（从保存的结果文件读取）
    generate_plots()

def generate_plots():
    output_dir = "output"
    import os, json

    # 实验1
    exp1_dir = os.path.join(output_dir, "experiment1")
    if os.path.exists(exp1_dir):
        exp1_results = {}
        for fname in os.listdir(exp1_dir):
            if fname.endswith("_random_results.json"):
                task_name = fname.replace("_random_results.json", "")
                with open(os.path.join(exp1_dir, fname), "r") as f:
                    data = json.load(f)
                exp1_results[task_name] = data
        if exp1_results:
            from visualization import plot_exp1_bar
            plot_exp1_bar(exp1_results, os.path.join(exp1_dir, "exp1_comparison.png"))

    # 实验2
    exp2_dir = os.path.join(output_dir, "experiment2")
    if os.path.exists(exp2_dir):
        exp2_results = {}
        for fname in os.listdir(exp2_dir):
            if fname.endswith("_step_impact_results.json"):
                task_name = fname.replace("_step_impact_results.json", "")
                with open(os.path.join(exp2_dir, fname), "r") as f:
                    data = json.load(f)
                exp2_results[task_name] = data
        if exp2_results:
            from visualization import plot_exp2_bar
            plot_exp2_bar(exp2_results, exp2_dir)

    # 实验3
    exp3_dir = os.path.join(output_dir, "experiment3")
    if os.path.exists(exp3_dir):
        exp3_results = {}
        for fname in os.listdir(exp3_dir):
            if fname.endswith("_shot_count_results.json"):
                task_name = fname.replace("_shot_count_results.json", "")
                with open(os.path.join(exp3_dir, fname), "r") as f:
                    data = json.load(f)
                exp3_results[task_name] = data
        if exp3_results:
            from visualization import plot_exp3_bar
            plot_exp3_bar(exp3_results, exp3_dir)
                    
                    
if __name__ == "__main__":
    main()