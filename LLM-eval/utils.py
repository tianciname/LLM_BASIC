import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def plot_radar_chart(scores_dict, output_path="llm_eval_radar.png", title="LLM Multi-Dimensional Evaluation"):
    """
    绘制雷达图（支持单模型或多模型对比）
    
    参数:
        scores_dict: 
            - 单模型: {"维度1": 分数, "维度2": 分数, ...}
            - 多模型: {"模型名": {"维度1": 分数, ...}, "模型名2": {...}}
        output_path: 输出图片路径
        title: 图表标题
    """
    # 自动检测输入格式，统一转换为多模型格式
    if isinstance(list(scores_dict.values())[0], dict):
        # 多模型格式
        model_names = list(scores_dict.keys())
        # 获取所有维度的并集（顺序一致）
        all_labels = list(scores_dict[model_names[0]].keys())
        data = {name: [scores_dict[name][label] for label in all_labels] 
                for name in model_names}
    else:
        # 单模型格式，包装成多模型
        model_names = ["Model"]
        all_labels = list(scores_dict.keys())
        data = {model_names[0]: list(scores_dict.values())}
    
    num_vars = len(all_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 设置美观的颜色循环
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # 绘制每个模型
    for model_name, values in data.items():
        values_closed = values + values[:1]
        color = next(colors)
        # 绘制填充区域（半透明）
        ax.fill(angles, values_closed, color=color, alpha=0.15)
        # 绘制线条
        ax.plot(angles, values_closed, color=color, linewidth=2, label=model_name)
        # 可选：在数据点上添加标记
        ax.scatter(angles[:-1], values, color=color, s=30, zorder=5)
    
    # 设置维度标签（调整位置避免重叠）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_labels, fontsize=11, fontweight='bold')
    # 对长标签旋转一下（可选）
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if len(label.get_text()) > 8:
            label.set_rotation(angle * 180 / np.pi - 90)
            label.set_horizontalalignment('center')
    
    # 设置径向轴（0~100，每20一格）
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize=9, color='gray')
    # 添加径向网格线（更淡雅）
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 去掉默认的极坐标边框的外侧圆环（可选，保持简洁）
    ax.spines['polar'].set_visible(False)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=10, framealpha=0.9)
    
    # 添加标题
    plt.title(title, fontsize=16, fontweight='bold', pad=30, color='#333333')
    
    # 调整底部边距，防止标签被裁剪
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"雷达图已保存至: {output_path}")

def generate_markdown_report(scores, filepath="evaluation_report.md"):
    """
    生成 Markdown 格式的评估报告（支持单/多模型）
    
    参数:
        scores: 与 plot_radar_chart 相同的格式
        filepath: 输出md文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 大语言模型 (LLM) 评估报告\n\n")
        
        # 自动检测格式
        if isinstance(list(scores.values())[0], dict):
            # 多模型
            model_names = list(scores.keys())
            f.write("## 各模型评估得分概览\n\n")
            for model in model_names:
                f.write(f"### {model}\n")
                f.write("| 评估维度 | 得分 (0-100) |\n")
                f.write("|---------|-------------|\n")
                for dim, val in scores[model].items():
                    f.write(f"| {dim} | {val:.2f} |\n")
                f.write("\n")
        else:
            # 单模型
            f.write("## 评估得分概览\n\n")
            f.write("| 评估维度 | 得分 (0-100) |\n")
            f.write("|---------|-------------|\n")
            for dim, val in scores.items():
                f.write(f"| {dim} | {val:.2f} |\n")
        
        f.write("\n## 可视化雷达图\n\n")
        f.write("![Radar Chart](./llm_eval_radar.png)\n")