import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_models():
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_model = PeftModel.from_pretrained(
        base_model,
        "../adapter/tinyllama-lora-adapter"
    )

    return base_model, lora_model, tokenizer


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
