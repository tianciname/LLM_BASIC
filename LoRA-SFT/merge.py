import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LoRAConfig, TrainingConfig
from injector import inject_lora
from LoRA import LoRALinear
def unload_lora(model):
    """
    遍历所有模块，将 LoRALinear 替换回 nn.Linear，并恢复合并后的权重。
    """
    replace_map = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            replace_map[name] = module

    for full_name, lora_module in replace_map.items():
        parent_name = ".".join(full_name.split(".")[:-1])
        attr_name = full_name.split(".")[-1]
        parent = model.get_submodule(parent_name)

        # 创建普通的 nn.Linear，权重直接使用 lora_module.linear（已经合并了增量）
        new_linear = nn.Linear(
            lora_module.in_features,
            lora_module.out_features,
            bias=lora_module.linear.bias is not None,
        )
        new_linear.weight.data = lora_module.linear.weight.data
        if lora_module.linear.bias is not None:
            new_linear.bias.data = lora_module.linear.bias.data

        # 替换
        setattr(parent, attr_name, new_linear)
    print(f"Unloaded LoRA from {len(replace_map)} layers.")

def merge_lora():

    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()

    # 1. 加载基座模型（float32）
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. 注入 LoRA 结构（空壳）
    model, _ = inject_lora(
        model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=0.0,
    )

    # 3. 加载训练好的 LoRA 增量
    adapter_state = torch.load(train_cfg.adapter_save_path, map_location="cpu")
    model.load_state_dict(adapter_state, strict=False)

    # 4. 合并权重（数学上把增量融进 linear）
    for module in model.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights()

    # 5. 【关键】把 LoRALinear 还原成 nn.Linear
    unload_lora(model)

    # 6. 保存干净的模型
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    model.save_pretrained("merged_model")
    tokenizer.save_pretrained("merged_model")
    print("Merged model saved to 'merged_model'")

if __name__ == "__main__":
    merge_lora()