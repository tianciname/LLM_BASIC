import re
import os
from model import generate_response

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _get_cache_path(task_name, strategy_name, sample_idx):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = f"{task_name}_{strategy_name}_{sample_idx}".replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.txt")


def _read_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _write_cache(cache_path, output):
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(output)


def normalize_text(text: str) -> str:
    return text.strip().lower()


def _strip_hallucinated_chat(text: str) -> str:
    # 1. 行首角色标记（多轮对话复述）
    cleaned = re.split(
        r'\n\s*(?:system|user|assistant|human)\b',
        text, maxsplit=1, flags=re.IGNORECASE
    )[0]
    # 2. 角色标记粘连在答案末尾（如 "Yesuser\n...", "$12.00.user\n..."）
    cleaned = re.split(
        r'(?<=\S)(?:system|user|assistant|human)(?:\s*:|\s*\n|\s*$)',
        cleaned, maxsplit=1, flags=re.IGNORECASE
    )[0]
    return cleaned.strip()


def extract_final_answer(output: str) -> str:
    """
    按提示词中规定的输出格式提取答案。
    Zero-shot/Structured → "Final Answer: <result>"
    Few-shot             → "So the answer is <result>"
    Direct               → 第一行
    """
    if not output:
        return ""

    output = _strip_hallucinated_chat(output)

    # 1. "Final Answer: <result>" — Zero-shot CoT / Structured CoT
    #    取最后一个不含 <result> 占位符的匹配（模型常会在思考过程中复述提示词指令）
    matches = list(re.finditer(
        r'Final\s+Answer\s*:\s*(.+?)(?:\.\s*$|\n|$)',
        output, re.IGNORECASE | re.MULTILINE
    ))
    for m in reversed(matches):
        ans = m.group(1).strip()
        if '<result>' not in ans:
            return ans

    # 2. "So the answer is <result>" — Few-shot CoT
    matches = list(re.finditer(
        r'So\s+the\s+answer\s+is\s+(.+?)(?:\.\s*$|\n|$)',
        output, re.IGNORECASE | re.MULTILINE
    ))
    for m in reversed(matches):
        ans = m.group(1).strip()
        if '<result>' not in ans:
            return ans

    # 3. "Answer:" 后面紧跟的内容 — Direct / 兜底（取第一个）
    m = re.search(r'Answer\s*:\s*(.+?)(?:\.\s*$|\n|$)', output, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # 4. 回退：取第一行（跳过 <think> 标签行）
    lines = [l.strip() for l in output.split('\n') if l.strip()]
    for line in lines:
        if line.lower() not in ('<think>', '</think>'):
            return line
    return lines[0] if lines else output.strip()


def check_answer_in_output(output: str, gold: str) -> bool:
    if not output or not gold:
        return False

    gold_lower = gold.lower().strip()
    cleaned = _strip_hallucinated_chat(output)

    # yes/no：检查首词（Direct），再回退到全文
    if gold_lower in ("yes", "no"):
        first_word = cleaned.strip().split()[0].lower().rstrip('.,;:!?')
        if first_word == gold_lower:
            return True

        # 在提取的答案中检查
        extracted = normalize_text(extract_final_answer(output))
        if re.search(r'\b' + gold_lower + r'\b', extracted):
            return True

        # 变体表达
        if gold_lower == "yes":
            return any(re.search(p, cleaned.lower()) for p in [
                r'\byou\s+can\b', r'\bit\s+is\s+(?:possible|true)\b',
                r'\babsolutely\b', r'\bindeed\b', r'\bcorrect\b', r'\bright\b',
            ])
        if gold_lower == "no":
            return any(re.search(p, cleaned.lower()) for p in [
                r'\byou\s+cannot\b', r"\byou\s+can't\b",
                r'\bnot\s+(?:possible|allowed|legal|true)\b',
                r'\bimpossible\b', r'\bwrong\b', r'\bfalse\b',
            ])
        return False

    # 数字 / 文本答案
    extracted = normalize_text(extract_final_answer(output))
    # 去除尾部标点（句子结束符、引号等，如 "40'" → "40"）
    extracted = extracted.strip('.,;:!?\'\"')

    # 数字容差匹配
    cleaned_num = re.sub(
        r'\$|€|£|¥|yuan|dollars|cents|meters|metres|cm|mm|km|kg|g|hours|hour|minutes|minute|seconds|%|，|,',
        '', extracted
    )
    try:
        gold_num = float(gold_lower.replace(',', '').replace('，', ''))
        numbers = re.findall(r'[\d]+\.?[\d]*', cleaned_num)
        for n in numbers:
            try:
                if abs(float(n) - gold_num) < 0.001:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    return gold_lower in extracted


def _has_valid_answer(output: str) -> bool:
    """检查模型输出是否包含有效答案（非占位符、非截断）"""
    if not output:
        return False
    output = _strip_hallucinated_chat(output)

    # 有效：包含真实内容的 "Final Answer:"
    matches = list(re.finditer(
        r'Final\s+Answer\s*:\s*(.+?)(?:\.\s*$|\n|$)',
        output, re.IGNORECASE | re.MULTILINE
    ))
    for m in reversed(matches):
        if '<result>' not in m.group(1):
            return True

    # 有效：包含真实内容的 "So the answer is"
    matches = list(re.finditer(
        r'So\s+the\s+answer\s+is\s+(.+?)(?:\.\s*$|\n|$)',
        output, re.IGNORECASE | re.MULTILINE
    ))
    for m in reversed(matches):
        if '<result>' not in m.group(1):
            return True

    # 有效：包含真实内容的 "Answer:"
    m = re.search(r'Answer\s*:\s*(.+?)(?:\.\s*$|\n|$)', output, re.IGNORECASE | re.MULTILINE)
    if m:
        ans = m.group(1).strip()
        if ans and '<result>' not in ans and '<think>' not in ans.lower():
            return True

    return False


def evaluate_single(samples, prompt_builder, model, tokenizer, model_type, log_file, prefix="", use_cache=True):
    correct = 0
    total = len(samples)
    valid_count = 0
    print(f"\n[{prefix}] 开始评估，共 {total} 个样本...")
    for idx, s in enumerate(samples):
        question = s["question"]
        gold = s["answer"]
        prompt = prompt_builder(question)

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{prefix}] 样本 {idx+1}/{total}")
        print(f"问题: {question}")
        print(f"标准答案: {gold}")
        print(f"提示词:\n{prompt}")

        cache_path = _get_cache_path(prefix, question[:40], idx) if use_cache else None
        cached = _read_cache(cache_path) if cache_path else None

        if cached is not None:
            response = cached
            print("[缓存命中]")
        else:
            try:
                response = generate_response(model, tokenizer, prompt, model_type)
                if not response:
                    response = "[空输出]"
            except Exception as e:
                response = f"[生成异常: {e}]"
            if cache_path:
                _write_cache(cache_path, response)

        is_correct = check_answer_in_output(response, gold)
        is_valid = _has_valid_answer(response)
        if is_correct:
            correct += 1
        if is_valid:
            valid_count += 1

        validity = "[有效回答]" if is_valid else "[无效回答]"
        status = "✓ 正确" if is_correct else "✗ 错误"

        # 控制台：只打印有效回答的完整输出；无效回答只打一行标记
        if is_valid:
            print(f"模型完整输出:\n{response}")
            print(f"是否正确: {status}")
        else:
            print(f"{validity} — 模型输出未包含有效答案（截断或仅含占位符），跳过完整打印")
        print(sep + "\n")

        # 日志文件：有效和无效回答都完整写入
        log_file.write(f"{sep}\n")
        log_file.write(f"{prefix} | 样本 {idx+1}: {validity} {status}\n")
        log_file.write(f"问题: {question}\n")
        log_file.write(f"标准答案: {gold}\n")
        log_file.write(f"模型完整输出:\n{response}\n")
        log_file.write(f"是否正确: {status}\n")
        log_file.write(f"{sep}\n\n")

    acc = correct / total if total else 0.0
    print(f"[{prefix}] 准确率: {acc:.2%} ({correct}/{total})  有效回答数: {valid_count}/{total}\n")
    log_file.write(f"{prefix} 整体准确率: {acc:.2%} ({correct}/{total})  有效回答数: {valid_count}/{total}\n")
    return acc
