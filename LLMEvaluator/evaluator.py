import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import json
from torch.utils.data import DataLoader
from tasks.glue_task import MNLITask
from tasks.squad_task import SQuADTask
from tasks.hellaswag_task import HellaSwagTask
from tasks.gsm8k_task import GSM8KTask
from tasks.humaneval_task import HumanEvalTask

class LLMEvaluator:
    def __init__(self, model_name_or_path, device="cuda", load_in_8bit=False):
        self.device = device
        # 加载 tokenizer 并设置 pad_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载因果语言模型（适用于 Qwen、Llama、GPT 等）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
            trust_remote_code=True
        )
        # 如果 device_map 为 None（CPU），手动移动到设备
        if device != "cuda" or self.model.device.type != "cuda":
            self.model.to(device)
        self.model.eval()
        self.results = {}

    def evaluate_task(self, task, split="validation", num_samples=None, batch_size=8):
        print(f"--- Evaluating {task.task_name} ({task.task_type}) ---")
        dataset = task.load_data(split, num_samples)
        if task.task_type == "classification":
            return self._eval_classification(task, dataset, batch_size)
        elif task.task_type == "multiple_choice":
            return self._eval_multiple_choice(task, dataset, batch_size)
        elif task.task_type == "generation":
            return self._eval_generation(task, dataset, batch_size)
        else:
            raise NotImplementedError

    def _eval_classification(self, task, dataset, batch_size):
        """
        零样本分类评估（使用因果 LM 计算选项的困惑度）。
        适用于 MNLI 等三项分类任务。
        """
        # 定义 MNLI 的三个选项标签及其对应文本
        label_words = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }
        
        all_preds = []
        all_labels = []
        total = len(dataset)
        
        # 提前 tokenize 选项文本，用于计算 loss
        option_tokens = {}
        for label, word in label_words.items():
            # 将选项词编码为输入 ID（不含前缀，只计算生成这个词的概率）
            tokenized = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
            option_tokens[label] = tokenized.input_ids[0]  # 1D tensor
        
        for idx in tqdm(range(total), desc="Evaluating MNLI"):
            ex = dataset[idx]
            # 提取 premise 和 hypothesis
            premise = ex['sentence1']
            hypothesis = ex['sentence2']
            true_label = ex['label']
            
            # 构建输入文本：premise + [SEP] + hypothesis
            input_text = f"{premise} {self.tokenizer.sep_token} {hypothesis}"
            # Tokenize 输入（不加选项）
            input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(self.device)
            
            # 计算每个选项的负对数似然
            best_label = None
            best_loss = float('inf')
            
            for label, opt_ids in option_tokens.items():
                # 将选项 ID 拼接到输入 ID 后面
                full_input_ids = torch.cat([input_ids[0], opt_ids.to(self.device)], dim=0).unsqueeze(0)
                # 标签位移：只计算选项部分的 loss（-100 忽略输入部分）
                labels = torch.full_like(full_input_ids, -100)
                # 选项部分的位置要设为真实 ID 以便计算 loss
                labels[0, input_ids.shape[1]:] = opt_ids.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(full_input_ids, labels=labels)
                    loss = outputs.loss.item()  # 平均负对数似然
                if loss < best_loss:
                    best_loss = loss
                    best_label = label
            
            all_preds.append(best_label)
            all_labels.append(true_label)
        
        # 计算准确率
        acc = sum(1 for p, t in zip(all_preds, all_labels) if p == t) / len(all_labels)
        return {"accuracy": acc}

    def _eval_multiple_choice(self, task, dataset, batch_size):
        if hasattr(task, "evaluate_model"):
            predictions, labels = task.evaluate_model(self.model, self.tokenizer, dataset, batch_size)
            metrics = task.compute_metrics(predictions, labels)
        else:
            raise NotImplementedError("Task must implement evaluate_model")
        return metrics

    def _eval_generation(self, task, dataset, batch_size):
        predictions = []
        references = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generation"):
            batch = dataset[i:i+batch_size]
            prompts = [task.get_prompt(ex) for ex in batch]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            # 解码完整输出
            gen_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            # 去掉原始 prompt 部分（简单按长度截取，更稳健的做法是使用 tokenizer 的 offset mapping，但这里用文本长度近似）
            for j, prompt in enumerate(prompts):
                gen_text = gen_texts[j]
                # 如果生成的文本以 prompt 开头，则去掉
                if gen_text.startswith(prompt):
                    gen_text = gen_text[len(prompt):].lstrip()
                predictions.append(gen_text)
            # 收集参考答案
            if "answer" in batch[0]:
                references.extend([ex["answer"] for ex in batch])
            else:
                references.extend(batch)
        metrics = task.compute_metrics(predictions, references)
        return metrics

    def run_all(self, tasks, num_samples=100, batch_size=8):
        for task in tasks:
            self.results[task.task_name] = self.evaluate_task(task, split="validation", num_samples=num_samples, batch_size=batch_size)
        return self.results

    def save_results(self, path="results.json"):
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)