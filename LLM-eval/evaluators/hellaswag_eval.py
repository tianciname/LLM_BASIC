from datasets import load_dataset
from tqdm import tqdm
import re
from config import CONFIG
def evaluate(pipeline, max_samples):
    """
    评估模型的常识推理与语言理解能力 (HellaSwag)
    """
    ds = load_dataset('Rowan/hellaswag', split='validation')
    sample_size = min(len(ds), max_samples)
    ds = ds.shuffle(seed=42).select(range(sample_size))
        
    correct = 0
    for item in tqdm(ds, desc="HellaSwag"):
        context = item['ctx']
        choices = item['endings']
        prompt = f"Context: {context}\nWhich ending is the most logical continuation?\n0: {choices[0]}\n1: {choices[1]}\n2: {choices[2]}\n3: {choices[3]}\n\nTask: Return ONLY the correct number (0, 1, 2, or 3)."
        
        messages = [{"role": "user", "content": prompt}]
        out = pipeline(messages, do_sample=False)[0]['generated_text']
        raw_response = out[-1]['content'].strip()
        
        # 剥离思维链草稿
        if "</think>" in raw_response:
            response = raw_response.split("</think>")[-1].strip()
        else:
            response = raw_response
            
        # 提取选项数字
        match = re.search(r'\b[0-3]\b', response)
        model_answer = match.group(0) if match else (response[0] if response else "")
        
        if model_answer == str(item['label']):
            correct += 1
            
    return (correct / sample_size) * 100 if sample_size > 0 else 0