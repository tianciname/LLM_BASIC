import re
import torch.nn as nn
from LoRA import LoRALinear

def print_model_structure(model: nn.Module):
    """打印模型结构概览（关键配置 + 模块层次）"""
    print("Model Structure Overview:")
    print("-" * 40)
    if hasattr(model, 'config'):
        config = model.config
        print(f"Model Type: {config.model_type}")
        print(f"Hidden Size: {config.hidden_size}")
        print(f"Num Layers: {config.num_hidden_layers}")
        print(f"Num Attention Heads: {config.num_attention_heads}")
        print(f"Vocab Size: {config.vocab_size}")
    # 递归打印模块树（深度限制3层，避免过长）
    def print_tree(module, indent=0, max_depth=3, prefix=""):
        if indent > max_depth:
            return
        children = list(module.named_children())
        if not children:
            print(f"{prefix}{module.__class__.__name__}")
        else:
            for name, child in children:
                child_str = f"{name}: {child.__class__.__name__}"
                print(f"{prefix}{child_str}")
                if indent < max_depth:
                    print_tree(child, indent+1, max_depth, prefix+"  ")
    print_tree(model)
    print("-" * 40)
    print()

def inject_lora(
    model: nn.Module,
    target_modules: list,
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
):
    """
    递归替换模型中的指定线性层为 LoRALinear。
    打印详细的注入信息并返回修改后的模型和可训练的 lora 参数列表。
    """
    # ---- 打印原始模型结构 ----
    print_model_structure(model)

    lora_params = []
    replace_map = {}
    # 1. 遍历所有模块，找出需要替换的线性层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            simple_name = name.split('.')[-1]
            if any(target in simple_name for target in target_modules):
                replace_map[name] = module

    total_replaced = len(replace_map)
    if total_replaced == 0:
        print("WARNING: No linear layers matched the target_modules. 请检查目标模块名称是否正确。")
        return model, lora_params

    print(f"共找到 {total_replaced} 个待替换的线性层：")
    print("-" * 60)

    # 2. 逐一替换
    for idx, (full_name, old_linear) in enumerate(replace_map.items(), start=1):
        parent_name = ".".join(full_name.split(".")[:-1])
        attr_name = full_name.split(".")[-1]
        parent = model.get_submodule(parent_name)

        new_layer = LoRALinear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=old_linear.bias is not None,
            original_linear=old_linear,
        )
        setattr(parent, attr_name, new_layer)
        lora_params.extend(new_layer.get_lora_params())

        # 打印每层信息
        print(f"[{idx}/{total_replaced}] Replaced {full_name}")
        print(f"      Original: ({old_linear.in_features}, {old_linear.out_features})")
        print(f"      LoRA: A({old_linear.in_features}, {r}) + B({r}, {old_linear.out_features})")
        print(f"      Device: {old_linear.weight.device}, dtype: {old_linear.weight.dtype}")

    # 3. 汇总统计
    print("-" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数:       {total_params:,}")
    print(f"可训练参数 (LoRA): {trainable_params:,}")
    print(f"可训练比例:       {trainable_params/total_params:.4%}")
    print("=" * 60)

    return model, lora_params


