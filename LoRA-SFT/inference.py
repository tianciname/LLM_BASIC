import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LoRAConfig, TrainingConfig
from injector import inject_lora

def inference_with_adapter(prompt):
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 注入 LoRA 结构
    model, _ = inject_lora(
        model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=0.0,
    )
    # 加载 adapter
    adapter_state = torch.load(train_cfg.adapter_save_path, map_location="cpu")
    model.load_state_dict(adapter_state, strict=False)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉 prompt 部分
    return response.split("<|im_start|>assistant\n")[-1].strip()

if __name__ == "__main__":
    print(inference_with_adapter("用三句话介绍深度学习"))


    