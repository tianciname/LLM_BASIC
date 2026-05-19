from typing import List, Dict


def build_direct_prompt(question: str) -> str:
    return (f"Answer the following question with just the final result.\n"
            f"Question: {question}\n"
            f"Answer:")


def build_zero_cot_prompt(question: str) -> str:
    return (f"Answer the following question.\n"
            f"Question: {question}\n"
            f"Answer: Let's think step by step. "
            f"End your response with 'Final Answer: <result>' on a separate line.\n")


def build_structured_cot_prompt(question: str) -> str:
    return (f"Answer the following question. You must first write out your reasoning steps clearly.\n"
            f"On the final line, output exactly 'Final Answer: <result>'.\n"
            f"Question: {question}\n"
            f"Answer:")


def build_few_shot_prompt(question: str, examples: List[Dict], k: int) -> str:
    selected = examples[:k]
    blocks = [
        "Answer the following questions. Think step by step before arriving at the answer.\n"
    ]
    for ex in selected:
        blocks.append(f"Question: {ex['question']}")
        blocks.append(f"Answer: {ex['rationale']} So the answer is {ex['answer']}.")
        blocks.append("")
    blocks.append(f"Now answer the next question. End with 'So the answer is <result>'.")
    blocks.append(f"Question: {question}")
    blocks.append(f"Answer:")
    return "\n".join(blocks)
