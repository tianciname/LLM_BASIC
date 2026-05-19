#!/usr/bin/env python
# train_distill.py — 知识蒸馏训练 (Teacher → Student)

import os
import gc
import torch
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from datasets import load_dataset
import logging
from config import *
from trainmon import TrainingMonitor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import setup_hf_token
setup_hf_token()

# -------------------- 日志净化：只保留本脚本的输出 --------------------
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
for name in ["httpx", "urllib3", "datasets", "transformers", "filelock", "fsspec"]:
    logging.getLogger(name).setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
# ----------------------------------------------------------------------


# -------------------- 蒸馏回调：桥接 Trainer → TrainingMonitor --------------------
class DistillationCallback(TrainerCallback):
    """将 Trainer 的 log 事件转发到 TrainingMonitor，附加 KL/CE 分量。"""

    def __init__(self, monitor: TrainingMonitor, trainer: "DistillationTrainer"):
        self.monitor = monitor
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 0 or logs is None:
            return

        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        if loss is None:
            return

        metrics = {
            "kl_loss": self.trainer.last_kl_val,
            "ce_loss": self.trainer.last_ce_val,
        }

        self.monitor.log(step=state.global_step, loss=loss, lr=lr, metrics=metrics)

    def on_train_end(self, args, state, control, **kwargs):
        pass


# -------------------- 蒸馏 Trainer：KL + CE 联合损失 --------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=ALPHA, temperature=TEMPERATURE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 暴露最近一次的 KL / CE 分量
        self.last_kl_val = None
        self.last_ce_val = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)

        # Teacher forward（无梯度，用完释放）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            del teacher_outputs

        # Student forward
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        if labels is not None:
            inputs["labels"] = labels

        # next-token prediction 对齐
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100).float()

        # 展平
        flat_s = shift_student.view(-1, shift_student.size(-1))
        flat_t = shift_teacher.view(-1, shift_teacher.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = mask.view(-1)

        valid = flat_mask.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            self.last_kl_val = 0.0
            self.last_ce_val = 0.0
            return (torch.tensor(0.0, device=student_logits.device, requires_grad=True),
                    student_outputs) if return_outputs else torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        s_valid = flat_s[valid]
        t_valid = flat_t[valid]
        l_valid = flat_labels[valid]

        # KL 散度
        kl_loss = F.kl_div(
            F.log_softmax(s_valid / self.temperature, dim=-1),
            F.softmax(t_valid / self.temperature, dim=-1),
            reduction='batchmean',
        ) * (self.temperature ** 2)

        # CE hard label loss
        ce_loss = F.cross_entropy(s_valid, l_valid, reduction='mean')

        # 存储分量供回调读取
        self.last_kl_val = kl_loss.item()
        self.last_ce_val = ce_loss.item()

        # 合并损失
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        del teacher_logits, student_logits
        return (loss, student_outputs) if return_outputs else loss


# -------------------- 数据预处理 --------------------
def format_instruction(example):
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n" + example["input"]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LENGTH):
    texts = []
    for i in range(len(examples["instruction"])):
        example = {
            "instruction": examples["instruction"][i],
            "input": examples.get("input", [""])[i] if "input" in examples else "",
            "output": examples["output"][i],
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
        desc="Tokenizing",
    )
    return dataset


# -------------------- 主函数 --------------------
def main():
    set_seed(SEED)

    # ── 初始化监控器 ──
    monitor = TrainingMonitor(log_dir=OUTPUT_DIR, experiment_name="distill")

    # ── 环境信息 ──
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        monitor.info(f"GPU: {torch.cuda.get_device_name(0)} | 总量 {total:.1f} GB")

    # ── 加载 Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 加载数据集 ──
    monitor.info(f"训练数据: {TRAIN_DATA_PATH}")
    train_dataset = load_and_prepare_dataset(TRAIN_DATA_PATH, tokenizer)
    eval_dataset = load_and_prepare_dataset(EVAL_DATA_PATH, tokenizer) if os.path.exists(EVAL_DATA_PATH) else None
    monitor.info(f"训练样本: {len(train_dataset)}" + (f" | 验证样本: {len(eval_dataset)}" if eval_dataset else ""))

    torch_dtype = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

    # ── 加载 Teacher ──
    monitor.info(f"加载 Teacher: {TEACHER_MODEL_NAME}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    teacher_model.config.use_cache = False
    teacher_model.gradient_checkpointing_enable()

    # ── 加载 Student ──
    monitor.info(f"加载 Student: {STUDENT_MODEL_NAME}")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    student_model.config.use_cache = False

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        monitor.info(f"双模型加载后显存占用: {used:.1f} GB")

    # ── Data Collator ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── 训练参数 ──
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
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_tqdm=True,
        report_to="none",
        log_level="error",
    )

    # ── Trainer + 回调 ──
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

    # 移除默认的 PrinterCallback/ProgressCallback，避免其打印原始 dict
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)

    callback = DistillationCallback(monitor, trainer)
    trainer.add_callback(callback)

    # ── 开始训练 ──
    monitor.info(f"配置: α={ALPHA} T={TEMPERATURE} | batch={BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS} | lr={LEARNING_RATE} | epochs={EPOCHS}")
    try:
        trainer.train()

        # ── 保存 ──
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        monitor.info(f"模型已保存至: {OUTPUT_DIR}")
    finally:
        del teacher_model
        del student_model
        torch.cuda.empty_cache()
        gc.collect()
        monitor.close()


if __name__ == "__main__":
    main()
