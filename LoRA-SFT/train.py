import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from config import LoRAConfig, TrainingConfig
from injector import inject_lora
from dataset import prepare_dataset

# 设置 HuggingFace Token（请替换为你自己的 Token，或通过环境变量传入）
os.environ["HF_TOKEN"] = "ssdadadadada"  # 已隐藏真实 token

def collate_fn(batch):
    """
    数据批次整理函数：对不同长度的序列进行 padding，
    并生成 attention_mask，同时将 labels 填充为 -100（忽略 loss 计算）。
    """
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]
    # padding
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != 0).long()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

def train():
    # ========== 阶段 1：解析配置 ==========
    print("\n" + "="*60)
    print("🚀 阶段 1/6：解析配置")
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()
    print("✅ 配置解析完成")
    print(f"   模型名称: {train_cfg.model_name}")
    print(f"   数据路径: {train_cfg.data_path}")
    print(f"   输出目录: {train_cfg.output_dir}")
    print("="*60)

    # ========== 阶段 2：加载分词器和基座模型 ==========
    print("\n" + "="*60)
    print("🚀 阶段 2/6：加载分词器和基座模型")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载完成")

    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        dtype=torch.bfloat16 if train_cfg.use_bfloat16 else torch.float32,
        trust_remote_code=True,
    ).to("cuda")
    model.config.use_cache = False  # 训练时必须关闭 KV 缓存
    print("✅ 基座模型加载完成并移至 GPU")
    print("="*60)

    # ========== 阶段 3：注入 LoRA 层 ==========
    print("\n" + "="*60)
    print("🚀 阶段 3/6：注入 LoRA 层")
    model, lora_params = inject_lora(
        model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
    )
    model.train()
    print("✅ LoRA 注入完成，模型已切换为训练模式")
    print("="*60)

    # ========== 阶段 4：准备数据集 ==========
    print("\n" + "="*60)
    print("🚀 阶段 4/6：准备数据集")
    dataset = prepare_dataset(train_cfg.data_path, tokenizer, max_length=train_cfg.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print("✅ 数据加载器创建完成")
    print(f"   批次大小: {train_cfg.per_device_batch_size}")
    print(f"   总批次数: {len(dataloader)}")
    print("="*60)

    # ========== 阶段 5：创建优化器和学习率调度器 ==========
    print("\n" + "="*60)
    print("🚀 阶段 5/6：创建优化器和学习率调度器")
    optimizer = torch.optim.AdamW(lora_params, lr=train_cfg.learning_rate)
    total_steps = len(dataloader) * train_cfg.num_epochs // train_cfg.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    print("✅ 优化器与调度器创建完成")
    print(f"   学习率: {train_cfg.learning_rate}")
    print(f"   预热步数: {train_cfg.warmup_steps}")
    print(f"   总训练步数: {total_steps}")
    print("="*60)

    # ========== 阶段 5.5：初始化训练监控器 ==========
    from trainmon import TrainingMonitor
    monitor = TrainingMonitor(log_dir=train_cfg.output_dir, experiment_name="lora_sft")
    print("📊 监控器已启动，TensorBoard 日志将保存至:", train_cfg.output_dir)

    # ========== 阶段 6：训练循环 ==========
    print("\n" + "="*60)
    print("🚀 阶段 6/6：开始训练循环")
    global_step = 0
    for epoch in range(train_cfg.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{train_cfg.num_epochs} 开始 ---")
        for step, batch in enumerate(dataloader):
            # 将数据移到模型所在设备
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 当前实际损失值（还原）
                current_loss = loss.item() * train_cfg.gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]

                # 记录到监控器（TensorBoard + CSV）
                monitor.log(step=global_step, loss=current_loss, lr=current_lr)

                if global_step % train_cfg.logging_steps == 0:
                    # 保留原有终端输出（可注释掉，因为 monitor 已经打印）
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")

                if global_step % train_cfg.save_steps == 0:
                    os.makedirs(train_cfg.output_dir, exist_ok=True)
                    torch.save(
                        {k: v for k, v in model.named_parameters() if v.requires_grad},
                        train_cfg.adapter_save_path,
                    )
                    print(f"💾 Adapter 已保存至 {train_cfg.adapter_save_path} (step {global_step})")

    # 最终保存
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    torch.save(
        {k: v for k, v in model.named_parameters() if v.requires_grad},
        train_cfg.adapter_save_path,
    )
    monitor.close()   # 关闭监控器
    print("\n🎉 训练完成，最终 Adapter 已保存。")
    print("="*60)

if __name__ == "__main__":
    train()
