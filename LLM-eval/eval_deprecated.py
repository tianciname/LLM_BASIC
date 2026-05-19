import os
import warnings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from config import CONFIG
from utils import plot_radar_chart, generate_markdown_report

# 显式导入各维度评估模块
import evaluators.mmlu_eval as mmlu_eval
import evaluators.hellaswag_eval as hellaswag_eval
import evaluators.humaneval_eval as humaneval_eval
import evaluators.truthfulqa_eval as truthfulqa_eval
import evaluators.safety_eval as safety_eval

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import setup_hf_token
setup_hf_token()
# 1. 屏蔽 Python 的所有 Deprecation 和 User 警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. 屏蔽 Hugging Face 自身的 logging 警告
os.environ["HF_MODULES_CACHE"] = "ignore" 
from transformers import logging
logging.set_verbosity_error()  # 只显示 Error，不显示 Warning
warnings.filterwarnings("ignore")

def main():
    print(f"[{CONFIG['device'].upper()}] 正在加载模型和分词器: {CONFIG['model_id']}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_id'], 
            device_map=CONFIG['device'],
            trust_remote_code=True,
            torch_dtype="auto"
        )
        # 初始化 text-generation pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    scores = {}
    max_s = CONFIG['max_samples']

    print("\n==============================================")
    print("           自动化多维评估任务启动             ")
    print("==============================================\n")
    
    # 逐一执行评估
    scores['Knowledge & Reasoning'] = mmlu_eval.evaluate(pipe, max_s)
    scores['Language Understanding'] = hellaswag_eval.evaluate(pipe, max_s)
    scores['Code Generation'] = humaneval_eval.evaluate(pipe, max_s)
    scores['Factuality'] = truthfulqa_eval.evaluate(pipe, max_s)
    scores['Safety'] = safety_eval.evaluate(pipe, max_s)

    print("\n==============================================")
    print("                 评估任务完成                 ")
    print("==============================================")
    for k, v in scores.items():
        print(f"{k}: {v:.2f}%")
        
    print("\n正在生成可视化图表及报告...")
    plot_radar_chart(scores, output_path="llm_eval_radar.png")
    generate_markdown_report(scores, filepath="evaluation_report.md")
    print("完成！结果已保存至 'llm_eval_radar.png' 与 'evaluation_report.md'")

if __name__ == "__main__":
    main()