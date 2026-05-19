import re
from model import generate_response

def normalize_text(text: str) -> str:
    return text.strip().lower()

def check_answer_in_output(output: str, gold: str) -> bool:
    """
    检查标准答案（gold）是否出现在模型输出中。
    - 若 gold 是数字，匹配数字（允许 $ 或 cm 等单位前缀/后缀）
    - 若 gold 是 yes/no，匹配完整单词 yes 或 no
    """
    if not output or not gold:
        return False
    output_lower = output.lower()
    gold_lower = gold.lower().strip()
    if gold_lower in ("yes", "no"):
        # 匹配完整单词 yes 或 no
        return bool(re.search(r'\b' + gold_lower + r'\b', output_lower))
    else:
        # 匹配数字（包括带单位的如 $40, 25 cm 等）
        # 构建正则：允许数字前后有 $、空格、逗号等
        pattern = re.escape(gold_lower)
        return bool(re.search(pattern, output_lower))

def evaluate_single(samples, prompt_builder, model, tokenizer, model_type, log_file, prefix=""):
    correct = 0
    total = len(samples)
    print(f"\n[{prefix}] 开始评估，共 {total} 个样本...")
    for idx, s in enumerate(samples):
        question = s["question"]
        gold = s["answer"]
        prompt = prompt_builder(question)

        # ===== 打印分隔线 =====
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{prefix}] 样本 {idx+1}/{total}")
        print(f"问题: {question}")
        print(f"标准答案: {gold}")
        print(f"提示词:\n{prompt}")
        # 生成
        try:
            response = generate_response(model, tokenizer, prompt, model_type)
            if not response:
                response = "[空输出]"
        except Exception as e:
            response = f"[生成异常: {e}]"

        # 直接使用模型输出作为“模型回答”，思维链即为输出本身（无额外提取）
        is_correct = check_answer_in_output(response, gold)
        if is_correct:
            correct += 1

        status = "✓ 正确" if is_correct else "✗ 错误"

        print(f"模型完整输出:\n{response}")
        print(f"是否正确: {status}")
        print(sep + "\n")

        # 写入日志文件
        log_file.write(f"{sep}\n")
        log_file.write(f"{prefix} | 样本 {idx+1}: {status}\n")
        log_file.write(f"问题: {question}\n")
        log_file.write(f"标准答案: {gold}\n")
        log_file.write(f"模型完整输出:\n{response}\n")
        log_file.write(f"是否正确: {status}\n")
        log_file.write(f"{sep}\n\n")

    acc = correct / total if total else 0.0
    print(f"[{prefix}] 准确率: {acc:.2%} ({correct}/{total})\n")
    log_file.write(f"{prefix} 整体准确率: {acc:.2%} ({correct}/{total})\n")
    return acc