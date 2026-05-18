from typing import List, Dict

def build_direct_prompt(question: str) -> str:
    """Direct answering without CoT."""
    return f"Q: {question}\nA:"

def build_zero_cot_prompt(question: str) -> str:
    """Zero-shot Chain-of-Thought."""
    return f"Q: {question}\nA: Let's think step by step."

def build_few_shot_prompt(question: str, examples: List[Dict], k: int) -> str:
    """Few-shot Chain-of-Thought with given examples."""
    selected = examples[:k]
    blocks = []
    for ex in selected:
        blocks.append(f"Q: {ex['question']}")
        blocks.append(f"A: {ex['rationale']} So the answer is {ex['answer']}.")
    blocks.append(f"Q: {question}")
    blocks.append("A:")
    return "\n".join(blocks)

def build_structured_cot_prompt(question: str) -> str:
    """Structured CoT: force reasoning then answer."""
    return (f"Q: {question}\n"
            f"Please provide your reasoning step by step, then give the final answer after 'Answer:'")