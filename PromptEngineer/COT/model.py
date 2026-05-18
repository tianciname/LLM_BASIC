import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from config import MODEL_NAME, DEVICE

def load_model_and_tokenizer():
    print(f"正在加载模型：{MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 判断模型类型
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model_type = "causal"
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_type == "causal":
        tokenizer.padding_side = "left"
    model.to(DEVICE)
    model.eval()
    print(f"模型加载完成，类型：{model_type}，设备：{DEVICE}")
    return model, tokenizer, model_type

def generate_response(model, tokenizer, prompt, model_type, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        if model_type == "causal":
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        else:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()