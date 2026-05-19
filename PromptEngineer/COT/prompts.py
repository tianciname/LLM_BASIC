from typing import List, Dict

def build_direct_prompt(question: str) -> str:
    return (f"Answer the following question.\n"
            f"Question: {question}\n"
            f"Answer: ")

def build_zero_cot_prompt(question: str) -> str:
    return (f"Answer the following question.\n"
            f"Question: {question}\n"
            f"Let's think step by step.\n")

def build_few_shot_prompt(question: str, examples: List[Dict], k: int) -> str:
    selected = examples[:k]
    blocks = []
    for ex in selected:
        blocks.append(f"Q: {ex['question']}")
        blocks.append(f"A: {ex['rationale']} So the answer is {ex['answer']}.")
    blocks.append(f"Now answer the next question.\nQ: {question}\nA:")
    return "\n".join(blocks)

def build_structured_cot_prompt(question: str) -> str:
    return (f"Answer the following question. You must first write out your reasoning steps clearly, "
            f"then provide the final answer in the format 'Final Answer: <result>'.\n"
            f"Question: {question}\n")