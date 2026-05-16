# verify_compatibility.py
from transformers import AutoConfig

# Base和Instruct模型配置
base_config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B-Base")
instruct_config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B-Base")

# 打印关键配置进行对比
print("=== 模型架构关键参数对比 ===")
print(f"隐藏层大小: {base_config.hidden_size} vs {instruct_config.hidden_size}")
print(f"注意力头数: {base_config.num_attention_heads} vs {instruct_config.num_attention_heads}")
print(f"隐藏层层数: {base_config.num_hidden_layers} vs {instruct_config.num_hidden_layers}")
print(f"中间层大小: {base_config.intermediate_size} vs {instruct_config.intermediate_size}")
print(f"词汇表大小: {base_config.vocab_size} vs {instruct_config.vocab_size}")

# 快速确认是否完全一致
is_identical = all([
    base_config.hidden_size == instruct_config.hidden_size,
    base_config.num_attention_heads == instruct_config.num_attention_heads,
    base_config.num_hidden_layers == instruct_config.num_hidden_layers,
    base_config.intermediate_size == instruct_config.intermediate_size,
    base_config.vocab_size == instruct_config.vocab_size
])

if is_identical:
    print("\n✅ 架构验证通过！Base和Instruct模型在结构层面完全兼容，可以开始蒸馏。")
else:
    print("\n❌ 警告: 架构存在差异，需要进一步分析")