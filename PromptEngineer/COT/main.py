import random
import csv
import os
import sys
from datetime import datetime
import torch
from config import (
    RANDOM_SEED, DEVICE, EVAL_STEPS, FEW_SHOT_STEP_RANGE,
    RUN_MAIN_EXPERIMENT, RUN_STEP_VERIFICATION,
    MATH_SAMPLE_SIZE, COMMONSENSE_SAMPLE_SIZE, MODEL_NAME,
    MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE
)
from data_loader import load_few_shot_examples, load_real_dataset_samples, filter_samples_by_steps
from model import load_model_and_tokenizer
from prompts import (
    build_direct_prompt,
    build_zero_cot_prompt,
    build_few_shot_prompt,
    build_structured_cot_prompt
)
from evaluation import evaluate_task
from visualization import plot_results, plot_step_results, plot_step_verification

# 确保输出目录存在
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_log_file(experiment_name):
    """创建一个带时间戳的日志文件，返回文件对象和自定义 log_print 函数"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{experiment_name}_{timestamp}.txt")
    f = open(filename, "w", encoding="utf-8")
    def log_print(*args, **kwargs):
        # 同时输出到控制台和文件
        text = " ".join(str(arg) for arg in args)
        print(text, **kwargs)
        f.write(text + "\n")
    return f, log_print

def write_experiment_config(log_func):
    """将当前实验配置写入日志"""
    log_func("========== 实验配置 ==========")
    log_func(f"模型: {MODEL_NAME}")
    log_func(f"生成参数: max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, do_sample={DO_SAMPLE}")
    log_func(f"随机种子: {RANDOM_SEED}")
    log_func(f"测试集步数过滤: {EVAL_STEPS}")
    log_func(f"混合示例步数范围: {FEW_SHOT_STEP_RANGE}")
    log_func(f"算术样本抽样数: {MATH_SAMPLE_SIZE}")
    log_func(f"常识样本抽样数: {COMMONSENSE_SAMPLE_SIZE}")
    log_func("==============================\n")

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 加载少样本示例池
    few_shot_pool = load_few_shot_examples()
    # 加载真实数据集测试样本
    print("正在从真实数据集采样测试样本...")
    test_tasks_all = load_real_dataset_samples(
        math_size=MATH_SAMPLE_SIZE,
        commonsense_size=COMMONSENSE_SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )

    # 根据配置过滤测试样本
    if EVAL_STEPS is not None and (isinstance(EVAL_STEPS, int) or (isinstance(EVAL_STEPS, list) and len(EVAL_STEPS) > 0)):
        filtered_tasks = []
        for task_name, samples in test_tasks_all:
            filtered = filter_samples_by_steps(samples, EVAL_STEPS)
            if filtered:
                filtered_tasks.append((task_name, filtered))
        if not filtered_tasks:
            print("警告：过滤后无测试样本，使用全部样本。")
            filtered_tasks = test_tasks_all
        test_tasks = filtered_tasks
    else:
        test_tasks = test_tasks_all

    # 加载模型（只需一次）
    model, tokenizer, model_type = load_model_and_tokenizer()

    # ==================== 主实验 ====================
    if RUN_MAIN_EXPERIMENT:
        print("\n启动主实验...")
        f_main, log_main = create_log_file("main_experiment")
        write_experiment_config(log_main)

        # 准备候选示例池（根据步数范围过滤）
        if FEW_SHOT_STEP_RANGE is not None:
            candidate_pool = [ex for ex in few_shot_pool if ex.get("steps", 1) in FEW_SHOT_STEP_RANGE]
            if not candidate_pool:
                candidate_pool = few_shot_pool
        else:
            candidate_pool = few_shot_pool

        shuffled = candidate_pool.copy()
        random.shuffle(shuffled)

        config_2shot = shuffled[:2]
        config_4shot = shuffled[:4]
        config_6shot = shuffled[:6]
        config_4shot_diff = shuffled[6:10] if len(shuffled) >= 10 else shuffled[:4]

        pool_1step = [ex for ex in few_shot_pool if ex.get("steps", 1) == 1]
        pool_2step = [ex for ex in few_shot_pool if ex.get("steps", 1) == 2]
        pool_multistep = [ex for ex in few_shot_pool if ex.get("steps", 1) >= 3]
        k_few = 4
        config_1step = pool_1step[:k_few]
        config_2step = pool_2step[:k_few]
        config_multistep = pool_multistep[:k_few]

        strategies = [
            ("无 CoT", lambda q: build_direct_prompt(q)),
            ("零样本 CoT", lambda q: build_zero_cot_prompt(q)),
            ("结构化 CoT", lambda q: build_structured_cot_prompt(q)),
            ("混合 2-shot", lambda q: build_few_shot_prompt(q, config_2shot, len(config_2shot))),
            ("混合 4-shot B", lambda q: build_few_shot_prompt(q, config_4shot, len(config_4shot))),
            ("混合 6-shot C", lambda q: build_few_shot_prompt(q, config_6shot, len(config_6shot))),
            ("混合 4-shot D(不同)", lambda q: build_few_shot_prompt(q, config_4shot_diff, len(config_4shot_diff))),
            ("仅1步示例 4-shot", lambda q: build_few_shot_prompt(q, config_1step, len(config_1step))),
            ("仅2步示例 4-shot", lambda q: build_few_shot_prompt(q, config_2step, len(config_2step))),
            ("仅多步示例 4-shot", lambda q: build_few_shot_prompt(q, config_multistep, len(config_multistep))),
        ]

        results = {task_name: {} for task_name, _ in test_tasks}
        step_results = {task_name: {} for task_name, _ in test_tasks}

        for task_name, samples in test_tasks:
            log_main(f"\n--- 任务: {task_name} ---")
            for strat_name, builder in strategies:
                log_main(f"  策略: {strat_name}")
                overall_acc, step_acc = evaluate_task(
                    samples, builder, model, tokenizer, model_type,
                    task_name=task_name, strategy_name=strat_name,
                    log_func=log_main
                )
                results[task_name][strat_name] = overall_acc
                step_results[task_name][strat_name] = step_acc
                log_main(f"    总体准确率: {overall_acc:.2%}")
                for s, acc in sorted(step_acc.items()):
                    log_main(f"      - {s}步: {acc:.2%}")

        # 打印表格到日志
        strategy_names = [name for name, _ in strategies]
        header = ["任务"] + strategy_names
        row_format = "{:<15}" + "{:>15}" * len(strategy_names)
        log_main("\n" + "="*120)
        log_main("总体准确率对比表格")
        log_main(row_format.format(*header))
        log_main("-"*120)
        for task_name, _ in test_tasks:
            row = [task_name] + [f"{results[task_name][s]:.2%}" for s in strategy_names]
            log_main(row_format.format(*row))
        log_main("="*120)

        all_steps = set()
        for task_name, _ in test_tasks:
            for strat_name, _ in strategies:
                all_steps.update(step_results[task_name][strat_name].keys())
        all_steps = sorted(all_steps)
        if all_steps:
            log_main("\n按推理步数准确率对比表格")
            for task_name, _ in test_tasks:
                log_main(f"\n--- {task_name} ---")
                step_header = ["策略"] + [f"{s}步" for s in all_steps]
                step_row_fmt = "{:<20}" + "{:>10}" * len(all_steps)
                log_main(step_row_fmt.format(*step_header))
                log_main("-"*(20+10*len(all_steps)))
                for strat_name, _ in strategies:
                    row = [strat_name] + [f"{step_results[task_name][strat_name].get(s, 0.0):.2%}" for s in all_steps]
                    log_main(step_row_fmt.format(*row))
        log_main("="*120)

        # 保存 CSV 到 output/
        csv_path = os.path.join(OUTPUT_DIR, "main_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(header)
            for task_name, _ in test_tasks:
                writer.writerow([task_name] + [results[task_name][s] for s in strategy_names])
        log_main(f"主实验 CSV 已保存至 {csv_path}")

        # 绘制图表
        plot_results(results, strategy_names, os.path.join(OUTPUT_DIR, "cot_comparison.png"))
        plot_step_results(step_results, all_steps, strategy_names, os.path.join(OUTPUT_DIR, "cot_step_comparison.png"))

        f_main.close()
        print(f"主实验日志已保存至 output/")

    # ==================== 步数验证实验 ====================
    if RUN_STEP_VERIFICATION:
        print("\n启动步数验证实验...")
        f_ver, log_ver = create_log_file("step_verification_experiment")
        write_experiment_config(log_ver)

        pool_1step = [ex for ex in few_shot_pool if ex.get("steps", 1) == 1]
        pool_2step = [ex for ex in few_shot_pool if ex.get("steps", 1) == 2]
        pool_multistep = [ex for ex in few_shot_pool if ex.get("steps", 1) >= 3]

        if FEW_SHOT_STEP_RANGE is not None:
            candidate_pool = [ex for ex in few_shot_pool if ex.get("steps", 1) in FEW_SHOT_STEP_RANGE]
            if not candidate_pool:
                candidate_pool = few_shot_pool
        else:
            candidate_pool = few_shot_pool
        shuffled = candidate_pool.copy()
        random.shuffle(shuffled)
        config_mixed_4shot = shuffled[:4]

        k_few = 4
        config_1step = pool_1step[:k_few]
        config_2step = pool_2step[:k_few]
        config_multistep = pool_multistep[:k_few]

        step_shot_strategies = [
            ("混合示例 4-shot", lambda q: build_few_shot_prompt(q, config_mixed_4shot, len(config_mixed_4shot))),
            ("仅1步示例 4-shot", lambda q: build_few_shot_prompt(q, config_1step, len(config_1step))),
            ("仅2步示例 4-shot", lambda q: build_few_shot_prompt(q, config_2step, len(config_2step))),
            ("仅多步示例 4-shot", lambda q: build_few_shot_prompt(q, config_multistep, len(config_multistep))),
        ]

        step_groups = {"1步": [1], "2步": [2], "多步(3+)": [3,4,5]}
        verification_results = {strat_name: {} for strat_name, _ in step_shot_strategies}

        for task_name, all_samples in test_tasks_all:  # 使用未过滤的全部样本
            for group_label, steps in step_groups.items():
                filtered = filter_samples_by_steps(all_samples, steps)
                if not filtered:
                    continue
                log_ver(f"\n--- 任务: {task_name}, 测试步数: {group_label} ---")
                for strat_name, builder in step_shot_strategies:
                    acc, _ = evaluate_task(
                        filtered, builder, model, tokenizer, model_type,
                        task_name=task_name, strategy_name=strat_name,
                        log_func=log_ver
                    )
                    key = f"{task_name}-{group_label}"
                    verification_results[strat_name][key] = acc
                    log_ver(f"  {strat_name}: {acc:.2%}")

        # 打印验证结果表格
        ver_header = ["策略"] + sorted(verification_results[list(verification_results.keys())[0]].keys())
        row_fmt = "{:<20}" + "{:>15}" * (len(ver_header)-1)
        log_ver("\n步数验证准确率表格")
        log_ver(row_fmt.format(*ver_header))
        log_ver("-"*(20+15*(len(ver_header)-1)))
        for strat_name in verification_results:
            row = [strat_name] + [f"{verification_results[strat_name].get(k, 0.0):.2%}" for k in ver_header[1:]]
            log_ver(row_fmt.format(*row))
        log_ver("="*120)

        # 保存 CSV
        csv_path = os.path.join(OUTPUT_DIR, "step_verification.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(ver_header)
            for strat_name in verification_results:
                row = [strat_name] + [verification_results[strat_name].get(k, 0.0) for k in ver_header[1:]]
                writer.writerow(row)
        log_ver(f"步数验证 CSV 已保存至 {csv_path}")

        # 绘制验证图
        first_task = test_tasks_all[0][0]
        plot_data = {}
        for strat_name in verification_results:
            plot_data[strat_name] = {}
            for group_label in step_groups.keys():
                key = f"{first_task}-{group_label}"
                plot_data[strat_name][group_label] = verification_results[strat_name].get(key, 0.0)
        plot_step_verification(plot_data, list(step_groups.keys()),
                               list(verification_results.keys()),
                               os.path.join(OUTPUT_DIR, "step_verification.png"))

        f_ver.close()
        print("步数验证实验日志已保存至 output/")

if __name__ == "__main__":
    main()