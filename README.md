
# Parameter-Efficient Fine-Tuning of TinyLlama using LoRA (PEFT)

## 📌 Project Overview

This project demonstrates **parameter-efficient fine-tuning (PEFT)** of a small open-source language model using **LoRA (Low-Rank Adaptation)**.

Instead of full fine-tuning of all 1.1B parameters, this project:

- Freezes the base model weights  
- Injects trainable LoRA adapters into attention layers  
- Trains only a small fraction of parameters  
- Evaluates improvement over the base model using perplexity  

The objective was to measure how effectively LoRA can specialize a pretrained instruction model using a relatively small dataset.

---

## 🧠 Base Model

- **Model:** TinyLlama-1.1B-Chat  
- **Parameters:** ~1.1B  
- **Fine-Tuning Method:** LoRA (PEFT)  

LoRA was applied to: 

target_modules = ["q_proj", "v_proj"]

Only ~0.3–0.6% of total parameters were trained.

---

## ⚙️ LoRA Configuration

| Parameter | Value |
|------------|--------|
| r | 8 / 16 |
| lora_alpha | 2 × r |
| lora_dropout | 0.05 |
| bias | none |
| task_type | CAUSAL_LM |

---

## 📊 Dataset

- **Dataset:** Databricks Dolly 15k  
- **Subset Sizes Tested:** 1500 and 3000 samples  
- **Train/Validation Split:** 90/10  

Each sample was formatted as:

Each sample was formatted as:

Instruction:
...
Response:
...

---

## 🏋️ Training Setup

- Epochs: 3  
- Learning Rate: 2e-4  
- Effective Batch Size: 16 (via gradient accumulation)  
- Optimizer: AdamW  
- Mixed Precision: FP16  

---

## 📈 Results

| Model | Dataset Size | r | Epochs | Eval Loss | Perplexity |
|-------|-------------|---|--------|-----------|------------|
| Base TinyLlama | - | - | - | 2.124 | 8.37 |
| LoRA Fine-tuned | 1500 | 8 | 3 | 1.698 | 5.46 |
| LoRA Fine-tuned | 1500 | 16 | 3 | 1.698 | 5.44 |
| LoRA Fine-tuned | 3000 | 16 | 3 | 1.680 | 5.41 |

---

## 🔍 Analysis

### 1️⃣ Perplexity Reduction

The base model achieved:

> **Perplexity: 8.37**

After LoRA fine-tuning:

> **Perplexity: 5.41 – 5.46**

This represents approximately a:

> **35% reduction in perplexity**

This indicates improved alignment with the instruction-response distribution.

---

### 2️⃣ Effect of LoRA Rank (r)

Increasing `r` from 8 → 16 resulted in marginal improvement: 5.46 → 5.44

This suggests:

- r=8 already provides sufficient adaptation capacity.
- Additional rank increases lead to diminishing returns.
- The task does not require high-rank adaptation.

---

### 3️⃣ Effect of Dataset Size

Increasing dataset size from 1500 → 3000 resulted in: 5.44 → 5.41

This modest improvement suggests:

- TinyLlama-Chat is already instruction-aligned.
- Dolly dataset distribution is stylistically consistent.
- Performance plateaus once format alignment is learned.

---

## 🧠 Key Takeaways

- LoRA successfully specialized the model without full fine-tuning.
- Significant perplexity reduction was achieved with minimal trainable parameters.
- Increasing LoRA rank beyond 8 provided limited gains.
- Doubling dataset size produced diminishing returns.
- PEFT is highly effective for lightweight model adaptation.

---

## 🚀 Why This Matters

This project demonstrates:

- Practical implementation of PEFT (LoRA)
- Controlled experiment design
- Quantitative evaluation using perplexity
- Hyperparameter comparison
- Analysis of diminishing returns

It highlights the ability to:

- Fine-tune LLMs efficiently
- Design reproducible ML experiments
- Interpret performance metrics critically
- Make informed architectural decisions

---

## 📂 Repository Structure

slm-lora-finetuning/
│
├── src/
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── results/
│ ├── experiment_table.md
│ └── metrics.json
│
├── adapter/
│ └── tinyllama-lora-adapter/
│
├── requirements.txt
└── README.m

---

## 🔮 Future Improvements

- Extend LoRA to additional attention modules (`k_proj`, `o_proj`)
- Fine-tune on domain-specific dataset (e.g., ML Q&A)
- Add qualitative output comparison section
- Compare LoRA vs full fine-tuning efficiency
- Deploy inference demo

---