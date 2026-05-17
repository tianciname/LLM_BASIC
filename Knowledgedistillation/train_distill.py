#!/usr/bin/env python
# train_distill.py

import os
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainerCallback,
)
from datasets import load_dataset
import logging
from config import *

os.environ["HF_TOKEN"] = "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"

# -------------------- 日志净化配置 --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error() 
# ------------------------------------------------------

try:
    from trainmon import TrainingMonitor
    TRAINMON_AVAILABLE = True
except ImportError:
    TRAINMON_AVAILABLE = False

# -------------------- 自定义回调：接管日志输出 --------------------
class TrainMonCallback(TrainerCallback):
    def __init__(self, monitor):
        self.monitor = monitor

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 0 and logs:
            loss = logs.get('loss', None)
            lr = logs.get('learning_rate', None)
            if loss is not None:
                if self.monitor:
                    try: self.monitor.log(step=state.global_step, loss=loss, lr=lr)
                    except: pass
                max_steps = state.max_steps
                print(f"[进度: {state.global_step}/{max_steps}] Loss: {loss:.4f} | LR: {lr:.2e}")

    def on_train_end(self, args, state, control, **kwargs):
        print("🎉 训练圆满结束！")

# -------------------- 自定义蒸馏Trainer --------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=ALPHA, temperature=TEMPERATURE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 彻底抽离 labels
        labels = inputs.pop("labels", None)
        
        # 1. 学生与教师模型前向传播
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        if labels is not None:
            inputs["labels"] = labels

        # 2. 准备错位对齐
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100).float()

        # 【显存核心优化】：分块（Chunk）计算 KL 散度与交叉熵，防止超大词表撑爆显存
        batch_size, seq_len, vocab_size = shift_student_logits.size()
        flat_student = shift_student_logits.view(-1, vocab_size)
        flat_teacher = shift_teacher_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = mask.view(-1)

        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_valid_tokens = flat_mask.sum() + 1e-8

        # 每次只计算 1024 个 token 的 Logits，显存开销直降到原本的几分之一
        chunk_size = 1024
        total_tokens = flat_student.size(0)

        for i in range(0, total_tokens, chunk_size):
            end_idx = min(i + chunk_size, total_tokens)
            
            s_chunk = flat_student[i:end_idx]
            t_chunk = flat_teacher[i:end_idx]
            l_chunk = flat_labels[i:end_idx]
            m_chunk = flat_mask[i:end_idx]

            if m_chunk.sum() == 0:
                continue

            # 当前块的 KL 散度
            kl_each = F.kl_div(
                F.log_softmax(s_chunk / self.temperature, dim=-1),
                F.softmax(t_chunk / self.temperature, dim=-1),
                reduction='none',
            ).sum(dim=-1)
            total_kl_loss += (kl_each * m_chunk).sum()

            # 当前块的交叉熵
            ce_each = F.cross_entropy(s_chunk, l_chunk, reduction='none', ignore_index=-100)
            total_ce_loss += (ce_each * m_chunk).sum()

        # 归一化损失
        kl_loss = (total_kl_loss / total_valid_tokens) * (self.temperature ** 2)
        ce_loss = total_ce_loss / total_valid_tokens

        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return (loss, student_outputs) if return_outputs else loss

# -------------------- 数据预处理 --------------------
def format_instruction(example):
    messages = []
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n" + example["input"]
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return messages

def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LENGTH):
    texts = []
    for i in range(len(examples["instruction"])):
        example = {
            "instruction": examples["instruction"][i],
            "input": examples.get("input", [""])[i] if "input" in examples else "",
            "output": examples["output"][i]
        }
        messages = format_instruction(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=max_length, return_tensors=None)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def load_and_prepare_dataset(data_path, tokenizer):
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    return dataset

# -------------------- 主训练函数 --------------------
def main():
    set_seed(SEED)
    logger.info("初始化环境与数据集...")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_and_prepare_dataset(TRAIN_DATA_PATH, tokenizer)
    eval_dataset = load_and_prepare_dataset(EVAL_DATA_PATH, tokenizer) if os.path.exists(EVAL_DATA_PATH) else None

    torch_dtype = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

    logger.info("加载模型到 GPU (静默加载中)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME, torch_dtype=torch_dtype, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME, torch_dtype=torch_dtype, trust_remote_code=True, device_map="auto", attn_implementation="sdpa"
    )
    student_model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=student_model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS, 
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=FP16,
        bf16=BF16,
        remove_unused_columns=False,
        seed=SEED,
        optim="adamw_torch_fused",
        
        dataloader_num_workers=4,       
        dataloader_pin_memory=True,     
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        disable_tqdm=True,      
        report_to="none",       
        log_level="error",      
    )

    monitor = None

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha=ALPHA,
        temperature=TEMPERATURE,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.add_callback(TrainMonCallback(monitor))

    logger.info("正式开始训练！...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()