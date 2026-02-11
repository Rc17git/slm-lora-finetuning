import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


def train_lora(dataset_size=1500, r=8, epochs=3):

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("databricks/databricks-dolly-15k")
    dataset = dataset["train"].shuffle(seed=42).select(range(dataset_size))

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    def format_prompt(example):
        if example["context"]:
            prompt = f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
        else:
            prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
        return {"text": prompt}

    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(
        ["instruction", "context", "response", "category", "text"]
    )
    eval_dataset = eval_dataset.remove_columns(
        ["instruction", "context", "response", "category", "text"]
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        save_strategy="no",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    results = trainer.evaluate()
    perplexity = math.exp(results["eval_loss"])

    model.save_pretrained("../adapter")
    tokenizer.save_pretrained("../adapter")

    print("Final Perplexity:", perplexity)

    return perplexity


if __name__ == "__main__":
    train_lora()
