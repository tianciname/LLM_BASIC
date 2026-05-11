import os
import shutil
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

def prepare_dataset(
    data_path,
    tokenizer,
    max_length=1024,
    save_dir="data/processed",
    overwrite=False,
):
    """
    Args:
        overwrite: 如果 save_dir 已存在，是否删除后重新生成
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

    # ================= 2. 数据清洗（可选） =================
    # ...

    # ================= 3. 分词与标签构造 =================
    print("\nStep 2: Tokenizing and constructing labels...")
    def tokenize_single(example):
        ins = example["instruction"]
        out = example["output"]
        if "input" in example and example["input"]:
            prompt = f"{ins}\n{example['input']}"
        else:
            prompt = ins
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text, truncation=True, max_length=max_length, padding=False
        )
        input_ids = tokenized["input_ids"]
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_start_ids = tokenizer.encode(assistant_start_str, add_special_tokens=False)
        assistant_idx = None
        for i in range(len(input_ids) - len(assistant_start_ids) + 1):
            if input_ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
                assistant_idx = i + len(assistant_start_ids)
                break
        labels = [-100] * len(input_ids)
        if assistant_idx is not None:
            labels[assistant_idx:] = input_ids[assistant_idx:]
        return {"input_ids": input_ids, "labels": labels}

    tokenized_dataset = raw_dataset.map(
        tokenize_single,
        remove_columns=raw_dataset.column_names,
    )
    print(f"Tokenized dataset created, samples: {len(tokenized_dataset)}")

    # ================= 4. 保存到磁盘 =================
    print(f"\nStep 3: Saving processed dataset to: {save_dir}")
    # 如果要求覆盖且目录已存在，先删除
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Removed existing directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(save_dir)
    print("Dataset saved successfully.")

    # ================= 5. 从磁盘重新加载 =================
    print(f"\nStep 4: Loading dataset from disk: {save_dir}")
    loaded_dataset = load_from_disk(save_dir)
    print(f"Loaded dataset samples: {len(loaded_dataset)}")
    print("=" * 50)
    return loaded_dataset