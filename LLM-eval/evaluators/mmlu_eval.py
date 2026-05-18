from datasets import load_dataset
from tqdm import tqdm
import re
from config import CONFIG
def evaluate(pipeline, max_samples):
    """
    评估模型的知识推理能力 (MMLU)
    """
    subsets = ['philosophy', 'nutrition', 'computer_security', 'high_school_physics']
    correct = 0
    total = 0
    
    for subset in subsets:
        ds = load_dataset('cais/mmlu', subset, split='test')
        sample_size = min(len(ds), max_samples // len(subsets))
        ds = ds.shuffle(seed=42).select(range(sample_size))
        
        for item in tqdm(ds, desc=f"MMLU - {subset}"):
            question = item['question']
            choices = item['choices']
            prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nTask: Choose the correct option (A, B, C, or D). Return ONLY the single letter."
            
            # 使用 Chat 模版结构，激活指令遵循状态
            messages = [{"role": "user", "content": prompt}]
            
            # 放开长度限制，让推理模型有充足空间写完思维链
            out = pipeline(messages,max_new_tokens=CONFIG["max_new_token_length"],max_length=None,do_sample=False)[0]['generated_text']
            raw_response = out[-1]['content'].strip()
            
            # 剥离思维链草稿 <think>...</think>
            if "</think>" in raw_response:
                response = raw_response.split("</think>")[-1].strip()
            else:
                response = raw_response
                
            response_upper = response.upper()
            correct_letter = chr(ord('A') + item['answer'])
            
            # 从过滤后的最终答案中精准搜索选项字母
            match = re.search(r'\b[A-D]\b', response_upper)
            model_answer = match.group(0) if match else (response_upper[0] if response_upper else "")
            
            if model_answer == correct_letter:
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0