#!/usr/bin/env python3
# run_eval.py
import argparse
import subprocess
import sys
import os
from pathlib import Path

def print_stage(msg):
    print("\n" + "=" * 60)
    print(f"🔎 {msg}")
    print("=" * 60)

def check_dependencies():
    try:
        import opencompass
    except ImportError:
        print("❌ OpenCompass 未安装，请运行: pip install opencompass")
        sys.exit(1)
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas 未安装，请运行: pip install pandas")
        sys.exit(1)
    try:
        import matplotlib
    except ImportError:
        print("❌ matplotlib 未安装，请运行: pip install matplotlib")
        sys.exit(1)

def download_core_data(data_dir="./opencompass_data"):
    data_path = Path(data_dir)
    if data_path.exists() and any(data_path.iterdir()):
        print("✅ 核心数据集已存在")
        return
    print("⚠️ 未发现核心数据集，正在下载...")
    import requests, zipfile, io
    url = "https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip"
    try:
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        print("✅ 数据集下载并解压完成")
    except Exception as e:
        print(f"❌ 下载失败: {e}，请手动下载并解压到 {data_dir}")
        sys.exit(1)

def generate_temp_config(args):
    """根据命令行参数动态生成临时配置文件"""
    config_content = f"""
# 临时评估配置（由 run_eval.py 自动生成）
from config import get_models, get_datasets, WORK_DIR, MAX_OUT_LEN

models = get_models()
datasets = get_datasets({args.datasets})

for d in datasets:
    if 'infer_cfg' in d and 'inferencer' in d['infer_cfg']:
        d['infer_cfg']['inferencer']['max_out_len'] = MAX_OUT_LEN

work_dir = WORK_DIR
"""
    temp_path = "temp_config.py"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    return temp_path

def main():
    parser = argparse.ArgumentParser(description="通用模型评估工具")
    parser.add_argument("--model_path", type=str, help="模型路径或HuggingFace ID")
    parser.add_argument("--tokenizer_path", type=str, help="分词器路径（默认与模型路径相同）")
    parser.add_argument("--model_type", choices=["chat", "base"], default="chat", help="模型类型")
    parser.add_argument("--model_abbr", type=str, default="eval_model", help="结果中显示的模型名称")
    parser.add_argument("--datasets", type=str, nargs="+", help="要评估的数据集，如: mmlu ceval gsm8k")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_out_len", type=int, default=1024)
    parser.add_argument("--work_dir", type=str, default="./outputs/eval_results")
    args = parser.parse_args()

    # 用命令行参数覆盖 config.py 中的默认值
    import config
    if args.model_path:
        config.MODEL_PATH = args.model_path
        config.TOKENIZER_PATH = args.tokenizer_path or args.model_path
    config.MODEL_TYPE = args.model_type
    config.MODEL_ABBR = args.model_abbr
    config.BATCH_SIZE = args.batch_size
    config.MAX_OUT_LEN = args.max_out_len
    config.WORK_DIR = args.work_dir
    datasets = args.datasets if args.datasets else config.DEFAULT_DATASETS

    print_stage("阶段 1/4：配置")
    print(f"模型路径    : {config.MODEL_PATH}")
    print(f"模型类型    : {config.MODEL_TYPE}")
    print(f"数据集      : {datasets}")
    print(f"批次大小    : {config.BATCH_SIZE}")
    print(f"输出目录    : {config.WORK_DIR}")

    # 检查依赖
    check_dependencies()

    # 下载/检查核心数据
    download_core_data()

    # 生成临时配置文件（使用命令行参数）
    temp_config = generate_temp_config(argparse.Namespace(datasets=datasets))

    # 阶段 2：推理
    print_stage("阶段 2/4：模型推理（Inference）")
    cmd_infer = [
        sys.executable, "-m", "opencompass.cli.main",
        temp_config,
        "-m", "infer",
        "--debug"
    ]
    print("执行推理...")
    ret = subprocess.run(cmd_infer)
    if ret.returncode != 0:
        print("❌ 推理阶段出错，请检查日志。")
        sys.exit(1)
    print("✅ 推理完成")

    # 阶段 3：评估
    print_stage("阶段 3/4：评估计分（Evaluation）")
    cmd_eval = [
        sys.executable, "-m", "opencompass.cli.main",
        temp_config,
        "-m", "eval",
        "--debug"
    ]
    print("执行评估...")
    ret = subprocess.run(cmd_eval)
    if ret.returncode != 0:
        print("❌ 评估阶段出错，请检查日志。")
        sys.exit(1)
    print("✅ 评估完成")

    # 阶段 4：可视化
    print_stage("阶段 4/4：结果可视化")
    result_dir = Path(config.WORK_DIR)
    summary_file = result_dir / "summary.csv"
    if not summary_file.exists():
        print(f"⚠️ 未找到汇总文件 {summary_file}，跳过可视化。")
        return
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_file)
    print("\n📊 模型得分：")
    print(df.to_string(index=False))

    # 绘图
    if 'dataset' in df.columns and 'score' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df['dataset'], df['score'], color='skyblue')
        plt.xlabel('Dataset')
        plt.ylabel('Score')
        plt.title(f"Evaluation Results - {config.MODEL_ABBR}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = result_dir / "evaluation_plot.png"
        plt.savefig(plot_path)
        print(f"📈 图表已保存至: {plot_path}")
        # 如果你在本地有图形界面，可取消注释下一行显示图片
        # plt.show()
    else:
        print("⚠️ summary.csv 列名不匹配，无法自动绘图。请手动查看。")

    # 清理临时配置
    os.remove(temp_config)
    print("\n🎉 评估全流程完成！")

if __name__ == "__main__":
    main()
