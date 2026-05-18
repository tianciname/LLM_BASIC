import re
import sys
from model import generate_response

def normalize_text(text: str) -> str:
    """Normalize text for exact match: lowercase, strip punctuation."""
    text = text.strip().lower()
    # Unify common boolean forms to yes/no
    if text in ("true", "yes", "y"):
        return "yes"
    if text in ("false", "no", "n"):
        return "no"
    # Remove trailing punctuation
    text = re.sub(r'[.!?,;:\'"]+$', '', text)
    return text.strip()

def extract_answer(generated_text: str) -> str:
    if not generated_text:
        return ""
    # Cut off any additional self-generated "Q:" or "问：" by the model
    for sep in ["Q:", "\nQ:", "问："]:
        if sep in generated_text:
            generated_text = generated_text.split(sep)[0]
            break

    # Try to match "answer is ..." or "答案是 ..."
    match = re.search(r'(?:answer is|答案是)\s*([^\n\.]+)', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Otherwise take the last non‑empty sentence
    sentences = re.split(r'[.\n]', generated_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        return sentences[-1]
    return generated_text.strip()

def exact_match(predicted: str, gold: str) -> bool:
    return normalize_text(predicted) == normalize_text(gold)

def evaluate_task(
    task_samples,
    prompt_builder,
    model,
    tokenizer,
    model_type,
    task_name="unknown",
    strategy_name="unknown",
    log_func=print
):
    correct = 0
    step_stats = {}   # {steps: [total, correct]}

    for idx, sample in enumerate(task_samples):
        question = sample["question"]
        gold = sample["answer"]
        steps = sample.get("steps", 1)

        try:
            prompt = prompt_builder(question)
            response = generate_response(model, tokenizer, prompt, model_type)
            pred_answer = extract_answer(response)
            is_correct = exact_match(pred_answer, gold)
            if is_correct:
                correct += 1

            step_stats.setdefault(steps, [0, 0])
            step_stats[steps][0] += 1
            if is_correct:
                step_stats[steps][1] += 1

            log_func(f"\n{'='*60}")
            log_func(f"Task: {task_name} | Strategy: {strategy_name} | Sample #{idx+1}")
            log_func(f"Question: {question}")
            log_func(f"Gold Answer: {gold}  (steps: {steps})")
            log_func(f"Prompt:\n{prompt}")
            log_func(f"Model Full Output:\n{response}")
            log_func(f"Extracted Answer: {pred_answer}")
            log_func(f"Correct: {'✓ Yes' if is_correct else '✗ No'}")
            log_func(f"{'='*60}")

        except Exception as e:
            log_func(f"Generation error (question: {question[:30]}...): {e}")

    total = len(task_samples)
    overall_acc = correct / total if total else 0.0
    step_acc = {s: corr/tot for s, (tot, corr) in step_stats.items()}
    return overall_acc, step_acc