import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，方便服务器使用
import seaborn as sns

# 定义每个任务的主指标映射
TASK_METRIC_MAP = {
    "mnli": "accuracy",
    "squad": "f1",          # 也可用 "exact_match"
    "hellaswag": "accuracy",
    "gsm8k": "accuracy",
    "humaneval": "pass@1"
}

# 指标显示名称
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "exact_match": "Exact Match",
    "pass@1": "Pass@1"
}

# 能力维度中文/英文标签
TASK_LABELS = {
    "mnli": "自然语言理解\n(NLU)",
    "squad": "阅读理解\n(RC)",
    "hellaswag": "常识推理\n(Commonsense)",
    "gsm8k": "数学推理\n(Math)",
    "humaneval": "代码生成\n(Code)"
}

def load_results(path):
    """加载单个模型结果JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_radar_values(results):
    """从结果字典提取各能力主指标数值，无结果则为0"""
    values = []
    labels = []
    for task_key, metric_key in TASK_METRIC_MAP.items():
        if task_key in results:
            metric_val = results[task_key].get(metric_key, 0)
            values.append(metric_val * 100)  # 转为百分比
        else:
            values.append(0)
        labels.append(TASK_LABELS.get(task_key, task_key))
    return labels, values

def plot_radar(labels, values, model_name, save_path="radar.png"):
    """绘制雷达图"""
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.plot(angles, values, color='steelblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title(f"模型能力雷达图: {model_name}", pad=20, fontsize=14, fontweight='bold')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存至 {save_path}")

def plot_bar_comparison(models_results, metric_key='accuracy', save_path="bar_comparison.png"):
    """
    多模型柱状图对比
    models_results: {模型名: results_dict} 字典
    metric_key: 要比较的指标，如 'accuracy'
    """
    task_names = list(TASK_METRIC_MAP.keys())
    model_names = list(models_results.keys())
    data = {task: [] for task in task_names}

    for model_name, results in models_results.items():
        for task in task_names:
            if task in results:
                val = results[task].get(metric_key, 0) * 100
            else:
                val = 0
            data[task].append(val)

    # 画图
    x = np.arange(len(task_names))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        offset = width * i - width * (len(model_names)-1)/2
        values = [data[task][i] for task in task_names]
        bars = ax.bar(x + offset, values, width, label=model_name)
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    display_metric = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
    ax.set_ylabel(f'{display_metric} (%)')
    ax.set_title(f'多模型 {display_metric} 对比')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], fontsize=10)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"柱状图已保存至 {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM评估结果可视化")
    parser.add_argument("--results", type=str, required=True, help="结果JSON文件路径")
    parser.add_argument("--model_name", type=str, default="MyModel", help="模型名称用于标题")
    parser.add_argument("--type", choices=["radar", "bar"], default="radar", help="图表类型")
    parser.add_argument("--compare", nargs="+", help="多模型对比：给出多个JSON文件路径")
    parser.add_argument("--metric", default="accuracy", help="柱状图对比指标")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    if args.type == "radar" and not args.compare:
        results = load_results(args.results)
        labels, values = extract_radar_values(results)
        out = args.output or f"{args.model_name}_radar.png"
        plot_radar(labels, values, args.model_name, out)

    elif args.type == "bar" and args.compare:
        models_data = {}
        for path in args.compare:
            model = Path(path).stem
            models_data[model] = load_results(path)
        out = args.output or f"comparison_{args.metric}.png"
        plot_bar_comparison(models_data, args.metric, out)
    else:
        print("雷达图请提供单个结果文件，柱状图对比请提供多个文件并指定 --compare")

if __name__ == "__main__":
    main()