import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_exp1_bar(results, save_path="exp1_comparison.png"):
    """
    实验1柱状图：两个任务在三个策略上的准确率。
    results: {task_name: {"Direct": acc, "Zero-shot CoT": acc, "Structured CoT": acc}}
    """
    strategies = ["Direct", "Zero-shot CoT", "Structured CoT"]
    task_names = list(results.keys())
    n_strategies = len(strategies)
    n_tasks = len(task_names)

    x = np.arange(n_strategies)
    width = 0.8 / n_tasks
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, task in enumerate(task_names):
        accs = [results[task][s] * 100 for s in strategies]
        bars = ax.bar(x + i * width, accs, width, label=task)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Experiment 1: CoT Strategy Comparison')
    ax.set_xticks(x + width * (n_tasks - 1) / 2)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"实验1柱状图已保存至 {save_path}")


def plot_exp2_bar(step_results, save_path_prefix="output/experiment2"):
    """
    实验2柱状图：x轴为示例步数，每步两个任务柱状图，Direct 基线为水平虚线。
    step_results: {task_name: {"baselines": {"Direct": acc, ...}, "few_shot_by_step": {step: acc}}}
    """
    import os
    for task_name, data in step_results.items():
        baselines = data.get("baselines", {})
        direct_acc = baselines.get("Direct", 0.0)
        few_shot = data.get("few_shot_by_step", {})
        if not few_shot:
            continue
        steps = sorted([int(k) for k in few_shot.keys()])
        accs = [few_shot[str(s)] * 100 for s in steps]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(steps))
        bars = ax.bar(x, accs, width=0.5, color='steelblue', label=f'{task_name} (4-shot)')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # Direct 基线
        ax.axhline(y=direct_acc * 100, color='red', linestyle='--', linewidth=1.5,
                   label=f'Direct baseline ({direct_acc:.1%})')

        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Reasoning steps in few-shot examples')
        ax.set_title(f'Experiment 2: Step Impact on {task_name}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}-step' for s in steps])
        ax.legend()
        ax.set_ylim(0, 105)
        plt.tight_layout()
        save_path = os.path.join(save_path_prefix, f"{task_name}_step_impact.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"实验2柱状图已保存至 {save_path}")


def plot_exp3_bar(shot_results, save_path_prefix="output/experiment3"):
    """
    实验3柱状图：每种示例步数一张图，x轴为shot数，柱状图为准确率，Direct基线为水平虚线。
    shot_results: {task_name: {"baselines": {"Direct": acc}, "by_step": {step: {shot: acc}}}}
    """
    import os
    for task_name, data in shot_results.items():
        baselines = data.get("baselines", {})
        direct_acc = baselines.get("Direct", 0.0)
        by_step = data.get("by_step", {})
        for step_str, shot_dict in by_step.items():
            step = int(step_str)
            shots = sorted([int(k) for k in shot_dict.keys()])
            accs = [shot_dict[str(k)] * 100 for k in shots]

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(shots))
            bars = ax.bar(x, accs, width=0.5, color='seagreen',
                          label=f'{task_name} ({step}-step examples)')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

            ax.axhline(y=direct_acc * 100, color='red', linestyle='--', linewidth=1.5,
                       label=f'Direct baseline ({direct_acc:.1%})')

            ax.set_ylabel('Accuracy (%)')
            ax.set_xlabel('Number of few-shot examples')
            ax.set_title(f'Experiment 3: Shot Count Impact ({step}-step examples) on {task_name}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{s} shots' for s in shots])
            ax.legend()
            ax.set_ylim(0, 105)
            plt.tight_layout()
            save_path = os.path.join(save_path_prefix, f"{task_name}_step{step}_shot_impact.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"实验3柱状图已保存至 {save_path}")