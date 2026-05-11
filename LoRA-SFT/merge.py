import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LoRAConfig, TrainingConfig
from injector import inject_lora

def merge_lora():
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch.float32,  # 合并时建议用 float32 避免精度损失
        device_map="auto",
        trust_remote_code=True,
    )
    # 注入相同的 LoRA 结构（暂时）
    model, _ = inject_lora(
        model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=0.0,
    )

    # 加载训练好的 adapter 权重
    adapter_state = torch.load(train_cfg.adapter_save_path, map_location="cpu")
    # 严格来说，我们只替换了部分模块，需要先将 adapter 权重载入模型
    model.load_state_dict(adapter_state, strict=False)

    # 合并所有权重
    for module in model.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights()

    # 移除 LoRA 结构，只保留原始 linear (通过替换回 nn.Linear 较麻烦，简单做法是保存合并后的模型)
    # 这里直接保存完整模型
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    model.save_pretrained("merged_model")
    tokenizer.save_pretrained("merged_model")
    print("Merged model saved to 'merged_model'")

if __name__ == "__main__":
    merge_lora()