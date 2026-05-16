import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from config import LoRAConfig, TrainingConfig
from injector import inject_lora
from dataset import prepare_dataset

# 设置 HuggingFace Token（请替换为你自己的 Token，或通过环境变量传入）
os.environ["HF_TOKEN"] = "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"

def train():

    # ========== 将 collate_fn 定义在函数内，确保能使用 tokenizer ==========
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
        # 因为打包后所有样本长度已固定为 max_length，理论上无填充，可直接全1；
        # 但为了保险，使用 tokenizer.pad_token_id 判断填充位置
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ========== 阶段 1：解析配置 ==========
    print("\n" + "="*60)
    print("🚀 阶段 1/6：解析配置")
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig()
    print("✅ 配置解析完成")
    print(f"   模型名称: {train_cfg.model_name}")
    print(f"   数据路径: {train_cfg.data_path}")
    print(f"   输出目录: {train_cfg.output_dir}")
    print(f"   验证集比例: {train_cfg.val_split}")
    print(f"   验证步数间隔: {train_cfg.val_steps}")
    print("="*60)

    # ========== 阶段 2：加载分词器和基座模型 ==========
    print("\n" + "=" * 60)
    print("🚀 阶段 2/6：加载分词器和基座模型")

    # 从模型 ID 提取简短名称（例如 "Qwen/Qwen3.5-0.8B-Base" → "Qwen3.5-0.8B-Base"）
    model_id = train_cfg.model_name
    model_name_short = model_id.split("/")[-1]   # 取最后一段作为文件夹名
    local_model_dir = os.path.join(train_cfg.models_path, model_name_short)

    # 检查本地是否已存在该模型
    if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
        print(f"🔍 发现本地模型 '{local_model_dir}'，直接从本地加载...")
        model_path = local_model_dir
    else:
        print(f"⬇️ 本地模型目录不存在，开始从 HuggingFace 下载 '{model_id}'...")
        # 下载模型与分词器（临时加载）
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if train_cfg.use_bfloat16 else torch.float32,
            trust_remote_code=True,
        )
        temp_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # 保存到本地
        os.makedirs(local_model_dir, exist_ok=True)
        temp_model.save_pretrained(local_model_dir)
        temp_tokenizer.save_pretrained(local_model_dir)
        print(f"✅ 模型已保存至 '{local_model_dir}'")
        # 释放临时模型占用的显存
        del temp_model
        del temp_tokenizer
        model_path = local_model_dir

    # 从本地加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载完成")

    # 从本地加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if train_cfg.use_bfloat16 else torch.float32,
        trust_remote_code=True,
    ).to("cuda")
    model.config.use_cache = False
    print("✅ 基座模型加载完成并移至 GPU")
    print("=" * 60)

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
    # 调用新的 prepare_dataset，返回训练集和验证集
    train_dataset, val_dataset = prepare_dataset(
        train_cfg.data_path,
        tokenizer,
        max_length=train_cfg.max_seq_length,
        val_split=train_cfg.val_split
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg.per_device_batch_size * 2,   # 验证时batch可稍大
            shuffle=False,
            collate_fn=collate_fn,
        )
    print("✅ 数据加载器创建完成")
    print(f"   训练集批次数: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   验证集批次数: {len(val_dataloader)}")
    print("="*60)

    # ========== 阶段 5：创建优化器和学习率调度器 ==========
    print("\n" + "="*60)
    print("🚀 阶段 5/6：创建优化器和学习率调度器")
    optimizer = torch.optim.AdamW(lora_params, lr=train_cfg.learning_rate)
    total_steps = len(train_dataloader) * train_cfg.num_epochs // train_cfg.gradient_accumulation_steps
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

    # ========== 阶段 6：训练循环（含验证） ==========
    print("\n" + "="*60)
    print("🚀 阶段 6/6：开始训练循环")
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(train_cfg.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{train_cfg.num_epochs} 开始 ---")
        model.train()
        for step, batch in enumerate(train_dataloader):
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
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")

                # ---------- 按步验证 ----------
                if val_dataloader is not None and train_cfg.val_steps > 0 and global_step % train_cfg.val_steps == 0:
                    model.eval()
                    val_loss_sum = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            device = next(model.parameters()).device
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            val_out = model(**val_batch)
                            val_loss_sum += val_out.loss.item()
                            val_batches += 1
                    avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
                    print(f"  📊 Step {global_step} 验证集 Loss: {avg_val_loss:.4f}")
                    monitor.writer.add_scalar("Loss/val", avg_val_loss, global_step)
                    # 保存最佳模型
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = train_cfg.adapter_save_path.replace('.pt', '_best.pt')
                        torch.save(
                            {k: v for k, v in model.named_parameters() if v.requires_grad},
                            best_path
                        )
                        print(f"  💾 最佳模型已更新 (val_loss={best_val_loss:.4f})")
                    model.train()  # 切回训练模式

                # ---------- 保存常规 adapter（与之前一致） ----------
                if global_step % train_cfg.save_steps == 0:
                    os.makedirs(train_cfg.output_dir, exist_ok=True)
                    torch.save(
                        {k: v for k, v in model.named_parameters() if v.requires_grad},
                        train_cfg.adapter_save_path,
                    )
                    print(f"💾 Adapter 已保存至 {train_cfg.adapter_save_path} (step {global_step})")

        # ---------- 按 epoch 验证（当 val_steps <= 0 时） ----------
        if val_dataloader is not None and train_cfg.val_steps <= 0:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    device = next(model.parameters()).device
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    val_out = model(**val_batch)
                    val_loss_sum += val_out.loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
            print(f"📊 Epoch {epoch+1} 验证集平均 Loss: {avg_val_loss:.4f}")
            monitor.writer.add_scalar("Loss/val", avg_val_loss, global_step)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = train_cfg.adapter_save_path.replace('.pt', '_best.pt')
                torch.save(
                    {k: v for k, v in model.named_parameters() if v.requires_grad},
                    best_path
                )
                print(f"  💾 最佳模型已更新 (val_loss={best_val_loss:.4f})")

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