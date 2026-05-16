import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def inference_with_merged_model(prompt, model_path="merged_model"):
    # 从合并后的模型目录加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # 或 torch.float16
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 构造对话消息
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取 assistant 的回复部分
    # 注意：不同模型可能使用不同的分隔符，Qwen 用 "<|im_start|>assistant\n"
    return response.split("<|im_start|>assistant\n")[-1].strip()

if __name__ == "__main__":
    print("===========================SFT之后：====================================")
    print(inference_with_merged_model("解释：李白乘舟江欲行"))
    print("===========================模型：====================================")
    print(inference_with_merged_model("解释：李白乘舟江欲行",model_path="/root/code/LLM_BASIC/LoRA-SFT/models/Qwen3.5-4B-Base"))