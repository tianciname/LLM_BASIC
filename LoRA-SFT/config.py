from dataclasses import dataclass, field
from typing import List

@dataclass
class LoRAConfig:
    r: int = 16                # LoRA 秩
    lora_alpha: int = 32       # 缩放系数
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3.5-4B-Base"   # 可换成真实模型
    data_path: str = "m-a-p/COIG-CQIA/chinese_traditional"
    output_dir: str = "outputs"
    adapter_save_path: str = "outputs/lora_adapter_best.pt"
    models_path: str = "models" #模型权重存放 
    val_split: float = 0.05       # 验证集比例，0 表示不划分验证集
    val_steps: int = 10         # 每隔多少 global_step 进行一次验证（若 <=0，则每个 epoch 结束时验证）
    per_device_batch_size: int = 6
    gradient_accumulation_steps: int = 4
    num_epochs: int = 6
    learning_rate: float = 2e-4
    max_seq_length: int = 128
    warmup_steps: int = 70
    logging_steps: int = 10
    save_steps: int = 500

    # 精度
    use_bfloat16: bool = True
    use_4bit: bool = False            # 自定义实现暂不包含量化，若需量化请用 bitsandbytes 加载模型