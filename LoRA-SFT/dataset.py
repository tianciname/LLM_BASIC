import os
import shutil
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer
from config import TrainingConfig

def save_dataset_to_txt(dataset, save_dir, tokenizer, filename="samples.txt", max_samples=None):
    """
    将数据集的 input_ids 解码为文本并保存为 txt 文件。
    max_samples: 最多保存多少条，None 表示全部保存。
    """
    path = os.path.join(save_dir, filename)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset:
            if max_samples is not None and count >= max_samples:
                break
            input_ids = sample["input_ids"]
            text = tokenizer.decode(input_ids, skip_special_tokens=False)
            f.write(text + "\n\n")   # 样本间用空行隔开
            count += 1
    print(f"Saved {count} samples to {path}")

def prepare_dataset(
    data_path,
    tokenizer,
    max_length=TrainingConfig.max_seq_length,
    save_dir="data/processed",
    overwrite=False,
    packing=True,           # 是否启用打包
    save_txt=True,          # 是否保存可读的 txt 文件
    val_split=0.0,          # 验证集比例，0 表示不划分
):
    """
    加载数据集 → 分词 → 可选打包 → 保存并重新加载。
    返回 (train_dataset, val_dataset)，val_dataset 可能为 None。
    """
    # ================= 1. 加载原始数据集 =================
    print("=" * 50)
    print("Step 1: Loading raw dataset...")
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        raw_dataset = load_dataset("json", data_files=data_path, split="train")
    elif '/' in data_path:
        parts = data_path.split('/')
        if len(parts) == 3:
            ds_id = '/'.join(parts[:2])
            subset = parts[2]
            raw_dataset = load_dataset(ds_id, subset, split="train")
        else:
            raise ValueError(f"data_path格式应为 'org/dataset/subset'，实际: {data_path}")
    else:
        raw_dataset = load_dataset(data_path, split="train")
    print(f"Raw dataset loaded: {data_path}")
    print(f"Number of raw samples: {len(raw_dataset)}")

    # ================= 2. 分词与标签构造 =================
    print("\nStep 2: Tokenizing and constructing labels...")
    def tokenize_single(example):
        """
        处理单条样本，将其分词并构造 labels。

        参数:
            example: dict, 数据集中一条原始样本，包含以下字段：
                - "instruction": str, 用户指令
                - "output": str, 期望的模型回复
                - "input": str (可选), 额外的输入上下文，若存在且非空则拼接到 instruction 后

        返回:
            dict: {
                "input_ids": list of int, 形状为 [seq_len]
                "labels": list of int, 形状为 [seq_len] (非 assistant 部分用 -100 填充，计算损失时被忽略)
            }
        """
        ins = example["instruction"]
        out = example["output"]
        # 如果存在 "input" 字段且非空，则拼接到 instruction 之后
        if "input" in example and example["input"]:
            prompt = f"{ins}\n{example['input']}"
        else:
            prompt = ins
        # 构造聊天模板消息 (list of dict)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out},
        ]
        # 应用聊天模板，生成格式化文本 (str)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # 分词，返回 dict，其中 "input_ids" 为 [seq_len] 的 int 列表
        tokenized = tokenizer(
            text, truncation=True, max_length=max_length, padding=False
        )
        input_ids = tokenized["input_ids"]          # list[int], 形状: [seq_len]

        # 定位 assistant 回复的起始 token 位置
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_start_ids = tokenizer.encode(assistant_start_str, add_special_tokens=False)
        # assistant_start_ids 为 list[int], 形状: [start_len]

        assistant_idx = None
        # 滑动窗口查找 assistant 起始标识符在 input_ids 中的位置
        for i in range(len(input_ids) - len(assistant_start_ids) + 1):
            if input_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
                assistant_idx = i + len(assistant_start_ids)    # int, assistant 回复开始的索引
                break

        # 构造 labels，初始全为 -100
        labels = [-100] * len(input_ids)           # list[int], 形状: [seq_len]
        if assistant_idx is not None:
            # 将 assistant 部分对应的 labels 替换为真实的 input_ids
            labels[assistant_idx:] = input_ids[assistant_idx:]   # 切片赋值，形状匹配

        return {"input_ids": input_ids, "labels": labels}

    tokenized_dataset = raw_dataset.map(
        tokenize_single,
        remove_columns=raw_dataset.column_names,
    )
    print(f"Tokenized dataset created, samples: {len(tokenized_dataset)}")

    # ================= 3. 打包（packing） =================
    if packing:
        print("\nStep 2.5: Packing tokenized sequences to max_length...")
        all_input_ids = []
        all_labels = []
        for sample in tokenized_dataset:
            all_input_ids.extend(sample["input_ids"])
            all_labels.extend(sample["labels"])

        packed_input_ids = []
        packed_labels = []
        for i in range(0, len(all_input_ids), max_length):
            chunk_input = all_input_ids[i:i+max_length]
            chunk_label = all_labels[i:i+max_length]
            if len(chunk_input) < max_length:
                continue
            packed_input_ids.append(chunk_input)
            packed_labels.append(chunk_label)

        packed_dataset = Dataset.from_dict({
            "input_ids": packed_input_ids,
            "labels": packed_labels,
        })
        print(f"Packed dataset created, samples: {len(packed_dataset)}")
        tokenized_dataset = packed_dataset

    # ================= 4. 划分训练集/验证集 =================
    if val_split > 0.0:
        print(f"\nStep 3: Splitting dataset (val ratio={val_split})")
        split_dataset = tokenized_dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    else:
        train_dataset = tokenized_dataset
        val_dataset = None

    # ================= 5. 保存到磁盘 =================
    print(f"\nStep 4: Saving processed dataset to: {save_dir}")
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Removed existing directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    train_dir = os.path.join(save_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    train_dataset.save_to_disk(train_dir)
    print(f"Train dataset saved to: {train_dir}")

    if val_dataset is not None:
        val_dir = os.path.join(save_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        val_dataset.save_to_disk(val_dir)
        print(f"Val dataset saved to: {val_dir}")

    # ================= 6. 保存可读的 txt 文件 =================
    if save_txt:
        print("\nStep 4.5: Saving human-readable txt files...")
        save_dataset_to_txt(train_dataset, train_dir, tokenizer)
        if val_dataset is not None:
            save_dataset_to_txt(val_dataset, val_dir, tokenizer)

    # ================= 7. 从磁盘重新加载 =================
    print(f"\nStep 5: Loading dataset from disk...")
    loaded_train = load_from_disk(train_dir)
    print(f"Loaded train samples: {len(loaded_train)}")

    loaded_val = None
    if val_dataset is not None:
        loaded_val = load_from_disk(val_dir)
        print(f"Loaded val samples: {len(loaded_val)}")

    print("=" * 50)
    return loaded_train, loaded_val