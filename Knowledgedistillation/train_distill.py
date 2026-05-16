#!/usr/bin/env python
# train_distill.py

import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset
import json
from typing import Dict, Optional
import argparse
from config import *
# -------------------- 自定义蒸馏Trainer --------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=ALPHA, temperature=TEMPERATURE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        # 冻结教师模型
        self.teacher_model.eval()
        # 确保教师模型与学生在同一设备上（Trainer会自动处理）
        # 但为了防止显存溢出，教师模型可以放在不同设备（如cpu）
        # 这里假设显存充足，默认与student同设备

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 学生前向（在 GPU）
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # 教师前向：输入移 CPU，输出移回 GPU
        with torch.no_grad():
            cpu_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    cpu_inputs[k] = v.cpu()
                else:
                    cpu_inputs[k] = v
            teacher_outputs = self.teacher_model(**cpu_inputs)
            teacher_logits = teacher_outputs.logits
            teacher_logits = teacher_logits.to(student_logits.device)

        # 后面的 KL 散度和交叉熵计算保持不变
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = inputs["labels"]
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100).float()

        kl_each = F.kl_div(
            F.log_softmax(shift_student_logits / self.temperature, dim=-1),
            F.softmax(shift_teacher_logits / self.temperature, dim=-1),
            reduction='none',
        ).sum(dim=-1)
        kl_loss = (kl_each * mask).sum() / (mask.sum() + 1e-8)
        kl_loss = kl_loss * (self.temperature ** 2)

        ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return (loss, student_outputs) if return_outputs else loss

# -------------------- 数据预处理 --------------------
def format_instruction(example):
    """将instruction-input-output转换为Qwen的对话格式"""
    # Qwen2.5使用chat template
    messages = []
    # 构造系统消息（可选）
    # messages.append({"role": "system", "content": "You are a helpful assistant."})
    # 用户消息
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n" + example["input"]
    messages.append({"role": "user", "content": user_content})
    # 助手回复
    messages.append({"role": "assistant", "content": example["output"]})
    return messages

def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LENGTH):
    """将数据集中的每个样本tokenize"""
    texts = []
    for i in range(len(examples["instruction"])):
        example = {
            "instruction": examples["instruction"][i],
            "input": examples.get("input", [""])[i] if "input" in examples else "",
            "output": examples["output"][i]
        }
        messages = format_instruction(example)
        # 应用chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    # 设置labels（用于计算交叉熵损失），与input_ids相同
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def load_and_prepare_dataset(data_path, tokenizer, is_train=True):
    """加载jsonl或json数据集"""
    # 支持json和jsonl
    if data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        dataset = load_dataset("json", data_files=data_path, split="train")
    # 映射预处理
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    return dataset

# -------------------- 主训练函数 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_NAME)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL_NAME)
    parser.add_argument("--train_file", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--eval_file", type=str, default=EVAL_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--fp16", action="store_true", default=FP16)
    parser.add_argument("--bf16", action="store_true", default=BF16)
    args = parser.parse_args()

    # 设置随机种子
    set_seed(SEED)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    print("Loading dataset...")
    train_dataset = load_and_prepare_dataset(args.train_file, tokenizer, is_train=True)
    eval_dataset = None
    if os.path.exists(args.eval_file):
        eval_dataset = load_and_prepare_dataset(args.eval_file, tokenizer, is_train=False)
        print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")

    # 加载模型
    print("Loading teacher model...")
    torch.cuda.empty_cache()

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",           # 教师模型放 CPU
        trust_remote_code=True,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 数据整理器（动态padding）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        padding=True,
        label_pad_token_id=-100,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )
    
    # 初始化蒸馏Trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("Starting distillation training...")
    trainer.train()

    # 保存最终模型
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()