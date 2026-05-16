from .base_task import BaseTask
from datasets import load_dataset
from evaluate import load

class SQuADTask(BaseTask):
    def __init__(self, num_samples=None):
        super().__init__("squad_v2", "question_answering")
        self.num_samples = num_samples
        self.metric = load("squad_v2")

    def load_data(self, split="validation", num_samples=None):
        dataset = load_dataset("squad_v2", split=split)
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset

    def preprocess(self, dataset, tokenizer, max_length=384, doc_stride=128):
        # 使用transformers官方QA预处理逻辑
        def prepare_features(examples):
            tokenized = tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            # 映射样本索引到原始数据
            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized.pop("offset_mapping")

            # 处理答案位置
            tokenized["start_positions"] = []
            tokenized["end_positions"] = []
            for i, offsets in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]
                answer = examples["answers"][sample_idx]
                # 如果没有答案，start=end=CLS (0)
                if len(answer["text"]) == 0:
                    tokenized["start_positions"].append(0)
                    tokenized["end_positions"].append(0)
                else:
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])
                    sequence_ids = tokenized.sequence_ids(i)

                    # 找到context的起止token
                    context_start = 0
                    while sequence_ids[context_start] != 1:
                        context_start += 1
                    context_end = len(sequence_ids) - 1
                    while sequence_ids[context_end] != 1:
                        context_end -= 1

                    # 如果答案超出范围，设为CLS
                    if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                        tokenized["start_positions"].append(0)
                        tokenized["end_positions"].append(0)
                    else:
                        # 找到对应token
                        idx = context_start
                        while idx <= context_end and offsets[idx][0] <= start_char:
                            idx += 1
                        start_pos = idx - 1
                        idx = context_end
                        while idx >= context_start and offsets[idx][1] >= end_char:
                            idx -= 1
                        end_pos = idx + 1
                        tokenized["start_positions"].append(start_pos)
                        tokenized["end_positions"].append(end_pos)
            return tokenized

        dataset = dataset.map(prepare_features, batched=True, remove_columns=dataset.column_names)
        return dataset

    def compute_metrics(self, predictions, references):
        # predictions: (start_logits, end_logits) 或直接 start/end indices
        # references: 原始answers (id, text)
        # 这里简化，接收后处理后的预测答案和参考答案
        return self.metric.compute(predictions=predictions, references=references)