# multi_model_eval.py
import os
import warnings
import gc
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from config import CONFIG  # 假设你原本的 config 包含 device 等
from utils import plot_radar_chart, generate_markdown_report

import evaluators.mmlu_eval as mmlu_eval
import evaluators.hellaswag_eval as hellaswag_eval
import evaluators.humaneval_eval as humaneval_eval
import evaluators.truthfulqa_eval as truthfulqa_eval
import evaluators.safety_eval as safety_eval

os.environ["HF_TOKEN"] = "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"

# 抑制警告（和你的原脚本一致）
os.environ["HF_TOKEN"] = "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"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_MODULES_CACHE"] = "ignore"
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


# ========== 多模型列表 ==========
MODELS_TO_EVAL = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]

# 每个模型的最大样本数（可统一，也可单独配置）
MAX_SAMPLES = CONFIG.get('max_samples', 200)   # 假设原config有该字段

def evaluate_single_model(model_id, device, max_samples):
    """加载单个模型并返回各维度得分字典"""
    print(f"\n{'='*50}")
    print(f"正在加载模型: {model_id}")
    print(f"{'='*50}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"模型加载失败 {model_id}: {e}")
        return None
    
    # 执行五个评估维度
    scores = {}
    print("\n[1/5] 运行 MMLU (知识推理)...")
    scores['Knowledge & Reasoning'] = mmlu_eval.evaluate(pipe, max_samples)
    
    print("[2/5] 运行 HellaSwag (语言理解)...")
    scores['Language Understanding'] = hellaswag_eval.evaluate(pipe, max_samples)
    
    print("[3/5] 运行 HumanEval (代码生成)...")
    scores['Code Generation'] = humaneval_eval.evaluate(pipe, max_samples)
    
    print("[4/5] 运行 TruthfulQA (事实性)...")
    scores['Factuality'] = truthfulqa_eval.evaluate(pipe, max_samples)
    
    print("[5/5] 运行 Safety (安全性)...")
    scores['Safety'] = safety_eval.evaluate(pipe, max_samples)
    
    # 清理显存，避免累积
    del pipe, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return scores

def main():
    device = CONFIG.get('device', 'cuda')
    all_models_scores = {}
    
    for model_id in MODELS_TO_EVAL:
        scores = evaluate_single_model(model_id, device, MAX_SAMPLES)
        if scores is not None:
            # 模型名取简短标识（也可以保留完整ID）
            short_name = model_id.split('/')[-1].replace('-Instruct', '')
            all_models_scores[short_name] = scores
            # 打印该模型结果
            print(f"\n>>> {short_name} 评估结果:")
            for k, v in scores.items():
                print(f"    {k}: {v:.2f}%")
    
    if len(all_models_scores) == 0:
        print("没有成功评估任何模型，退出。")
        return
    
    # 生成多模型对比雷达图
    print("\n正在生成多模型对比雷达图及报告...")
    plot_radar_chart(all_models_scores, output_path="multi_model_radar.png", 
                     title="LLM Multi-Model Comparison")
    generate_markdown_report(all_models_scores, filepath="multi_model_report.md")
    print("完成！结果已保存至 'multi_model_radar.png' 与 'multi_model_report.md'")

if __name__ == "__main__":
    main()