import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, strategy_names, save_path="cot_comparison.png"):
    """Plot overall accuracy comparison as grouped bar chart."""
    task_names = list(results.keys())
    n_tasks = len(task_names)
    n_strategies = len(strategy_names)
    x = np.arange(n_tasks)
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat in enumerate(strategy_names):
        acc_values = [results[task][strat] * 100 for task in task_names]
        bars = ax.bar(x + i * width, acc_values, width, label=strat)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparison of Different CoT Prompting Strategies')
    ax.set_xticks(x + width * (n_strategies - 1) / 2)
    ax.set_xticklabels(task_names)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Overall comparison chart saved to {save_path}")


def plot_step_results(step_results, all_steps, strategy_names, save_path="cot_step_comparison.png"):
    """Plot accuracy by reasoning step count for each task."""
    task_names = list(step_results.keys())
    n_tasks = len(task_names)
    n_steps = len(all_steps)
    n_strategies = len(strategy_names)

    fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5), squeeze=False)
    width = 0.8 / n_strategies

    for task_idx, (task_name, ax) in enumerate(zip(task_names, axes[0])):
        for i, strat in enumerate(strategy_names):
            accs = [step_results[task_name][strat].get(s, 0.0)*100 for s in all_steps]
            x_pos = [j + i*width for j in range(n_steps)]
            bars = ax.bar(x_pos, accs, width, label=strat)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=6)
        ax.set_title(f'{task_name} Accuracy by Reasoning Steps')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks([j + width*(n_strategies-1)/2 for j in range(n_steps)])
        ax.set_xticklabels([f'{s}-step' for s in all_steps])
        ax.set_ylim(0, 105)
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='x-small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Step-wise comparison chart saved to {save_path}")


def plot_step_verification(results_matrix, x_labels, strategy_names, save_path="step_verification.png"):
    """
    Plot step verification experiment: different few-shot step configurations
    evaluated on different test step groups.
    """
    n_strats = len(strategy_names)
    n_steps = len(x_labels)
    x = np.arange(n_steps)
    width = 0.8 / n_strats

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, strat in enumerate(strategy_names):
        accs = [results_matrix[strat].get(step, 0.0)*100 for step in x_labels]
        bars = ax.bar(x + i*width, accs, width, label=strat)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Sensitivity to Few-shot Example Steps vs Test Steps')
    ax.set_xticks(x + width*(n_strats-1)/2)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Step verification chart saved to {save_path}")