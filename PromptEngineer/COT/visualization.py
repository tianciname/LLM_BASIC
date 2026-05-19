import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 统一配色方案
COLOR_ARITHMETIC = '#E8795A'   # coral — 暖色，算术任务
COLOR_COMMONSENSE = '#5B9BD5'  # steel blue — 冷色，常识任务
COLOR_BASELINE = '#888888'     # gray — 基线

LINE_STYLE_ARITHMETIC = dict(color=COLOR_ARITHMETIC, marker='o', linewidth=2.2, markersize=8,
                             markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLOR_ARITHMETIC)
LINE_STYLE_COMMONSENSE = dict(color=COLOR_COMMONSENSE, marker='s', linewidth=2.2, markersize=8,
                              linestyle='--', markerfacecolor='white', markeredgewidth=2,
                              markeredgecolor=COLOR_COMMONSENSE)
BASELINE_STYLE = dict(color=COLOR_BASELINE, linestyle=':', linewidth=1.8, alpha=0.8)


def _style_ax(ax, title, xlabel, ylabel='Accuracy (%)'):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)


def _annotate_line(ax, x_vals, y_vals, style):
    """在线上的每个数据点标注数值。"""
    for x, y in zip(x_vals, y_vals):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold',
                    color=style['color'])


# ---- 实验1：CoT 类型对比 ----
def plot_exp1(results, save_path="exp1_comparison.png"):
    """
    折线图：两任务在三种策略上的准确率，加 Direct 基线水平线。
    results: {task_name: {"Direct": acc, "Zero-shot CoT": acc, "Structured CoT": acc}}
    """
    strategies = ["Direct", "Zero-shot CoT", "Structured CoT"]
    x = np.arange(len(strategies))

    fig, ax = plt.subplots(figsize=(10, 6))

    for task_name, style, label in [
        ("Arithmetic", LINE_STYLE_ARITHMETIC, "Arithmetic"),
        ("Commonsense", LINE_STYLE_COMMONSENSE, "Commonsense"),
    ]:
        if task_name not in results:
            continue
        data = results[task_name]
        accs = [data[s] * 100 for s in strategies]
        ax.plot(x, accs, **style, label=label, zorder=3)
        _annotate_line(ax, x, accs, style)

    # Direct 基线参考线（取两任务 Direct 的均值）
    direct_vals = [results[t]["Direct"] * 100 for t in results if "Direct" in results[t]]
    if direct_vals:
        avg = np.mean(direct_vals)
        ax.axhline(y=avg, **BASELINE_STYLE, label=f'Direct baseline ({avg:.1f}%)')

    _style_ax(ax, 'Experiment 1: CoT Strategy Comparison', 'Prompt Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[可视化] 实验1 图表已保存至 {save_path}")


# ---- 实验2：示例步数影响 ----
def plot_exp2(step_results, save_path="exp2_step_impact.png"):
    """
    折线图：x=示例步数，两任务各一条线，Direct 基线为水平虚线。
    step_results: {task_name: {"baselines": {"Direct": acc}, "few_shot_by_step": {step: acc}}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    all_steps = set()
    for data in step_results.values():
        all_steps.update(int(k) for k in data.get("few_shot_by_step", {}))
    all_steps = sorted(all_steps)
    if not all_steps:
        print("[可视化] 实验2 无数据，跳过")
        return

    for task_name, style, label in [
        ("Arithmetic", LINE_STYLE_ARITHMETIC, "Arithmetic (4-shot)"),
        ("Commonsense", LINE_STYLE_COMMONSENSE, "Commonsense (4-shot)"),
    ]:
        if task_name not in step_results:
            continue
        data = step_results[task_name]
        few_shot = data.get("few_shot_by_step", {})
        x_vals = [s for s in all_steps if str(s) in few_shot]
        y_vals = [few_shot[str(s)] * 100 for s in x_vals]
        if x_vals:
            ax.plot(x_vals, y_vals, **style, label=label, zorder=3)
            _annotate_line(ax, x_vals, y_vals, style)

    # Direct 基线
    direct_vals = [step_results[t]["baselines"].get("Direct", 0) * 100
                   for t in step_results if "baselines" in step_results[t]]
    if direct_vals:
        avg = np.mean(direct_vals)
        ax.axhline(y=avg, **BASELINE_STYLE, label=f'Direct baseline ({avg:.1f}%)')

    _style_ax(ax, 'Experiment 2: Impact of Few-Shot Reasoning Steps',
              'Reasoning Steps in Few-Shot Examples')
    ax.set_xticks(all_steps)
    ax.set_xticklabels([f'{s}-step' for s in all_steps], fontsize=11)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[可视化] 实验2 图表已保存至 {save_path}")


# ---- 实验3：提示词数量 × 步数 ----
def plot_exp3(shot_results, save_path="exp3_shot_impact.png"):
    """
    折线图：每种步数一张子图，x=shot 数，两任务各一条线 + Direct 基线。
    shot_results: {task_name: {"baselines": {"Direct": acc}, "by_step": {step: {shot: acc}}}}
    """
    # 收集所有 step
    all_steps = set()
    for data in shot_results.values():
        all_steps.update(int(k) for k in data.get("by_step", {}))
    all_steps = sorted(all_steps)
    if not all_steps:
        print("[可视化] 实验3 无数据，跳过")
        return

    n_steps = len(all_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(9 * n_steps, 6))
    if n_steps == 1:
        axes = [axes]

    for ax, step in zip(axes, all_steps):
        step_str = str(step)

        for task_name, style, label in [
            ("Arithmetic", LINE_STYLE_ARITHMETIC, "Arithmetic"),
            ("Commonsense", LINE_STYLE_COMMONSENSE, "Commonsense"),
        ]:
            if task_name not in shot_results:
                continue
            by_step = shot_results[task_name].get("by_step", {})
            shot_dict = by_step.get(step_str, {})
            if not shot_dict:
                continue
            shots = sorted(int(k) for k in shot_dict.keys())
            accs = [shot_dict[str(k)] * 100 for k in shots]
            ax.plot(shots, accs, **style, label=label, zorder=3)
            _annotate_line(ax, shots, accs, style)

        # Direct 基线
        direct_vals = [shot_results[t]["baselines"].get("Direct", 0) * 100
                       for t in shot_results if "baselines" in shot_results[t]]
        if direct_vals:
            avg = np.mean(direct_vals)
            ax.axhline(y=avg, **BASELINE_STYLE, label=f'Direct baseline ({avg:.1f}%)')

        _style_ax(ax, f'Exp3: Shot Count Impact ({step}-step examples)',
                  'Number of Few-Shot Examples')
        ax.set_xticks(sorted(int(k) for k in shot_dict.keys()))

    # 统一图例（取第一个子图的）
    handles, labels = axes[0].get_legend_handles_labels()
    # 去重
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               fontsize=10, ncol=len(by_label), framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Experiment 3: Few-Shot Count × Reasoning Step Interaction',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 实验3 图表已保存至 {save_path}")
