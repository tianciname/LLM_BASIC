import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, DEVICE, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE

def load_model_and_tokenizer():
    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(DEVICE)
    model.eval()
    model_type = "causal"
    print(f"Model loaded, type: {model_type}, device: {DEVICE}")
    return model, tokenizer, model_type

def generate_response(model, tokenizer, prompt, model_type, max_new_tokens=MAX_NEW_TOKENS):
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": DO_SAMPLE,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if DO_SAMPLE:
            gen_kwargs["temperature"] = TEMPERATURE
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return ""