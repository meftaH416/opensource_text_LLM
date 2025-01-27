# -*- coding: utf-8 -*-
""" 
Phi-3-mini (3.8B) Fine-Tuning with QLoRA for IDF Generation
"""
# Install required libraries
!pip install -q -U transformers peft accelerate trl bitsandbytes datasets

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# --------------------------------------------------------
# 1. Load Dataset (Replace with your IDF dataset)
# --------------------------------------------------------
# Sample dataset format: [{"input": "instruction", "output": "IDF syntax"}]
dataset = load_dataset("json", data_files="idf_dataset.json", split="train")

# --------------------------------------------------------
# 2. Configure Model & Tokenizer (4-bit QLoRA)
# --------------------------------------------------------
model_id = "microsoft/Phi-3-mini-4k-instruct"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for QLoRA training
model = prepare_model_for_kbit_training(model)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# --------------------------------------------------------
# 3. Configure LoRA Adapters
# --------------------------------------------------------
peft_config = LoraConfig(
    r=8,                  # LoRA rank
    lora_alpha=32,        # Alpha scaling
    target_modules=["Wqkv", "out_proj"],  # Phi-3 specific layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --------------------------------------------------------
# 4. Training Arguments
# --------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./phi3-idf-results",
    num_train_epochs=3,
    per_device_train_batch_size=2,     # Reduce if OOM
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="no",
    report_to="none"
)

# --------------------------------------------------------
# 5. Initialize Trainer with Instruction Format
# --------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=1024,               # Matches Phi-3's context window
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=lambda x: (
        f"<|user|>\nGenerate IDF for: {x['input']}"
        f"<|assistant|>\n{x['output']}"
    )
)

# --------------------------------------------------------
# 6. Train & Save
# --------------------------------------------------------
trainer.train()

# Save adapter
trainer.model.save_pretrained("phi3-mini-idf-lora")
tokenizer.save_pretrained("phi3-mini-idf-lora")

# --------------------------------------------------------
# 7. Inference with Fine-Tuned Model
# --------------------------------------------------------
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

# Merge LoRA adapter
model = PeftModel.from_pretrained(base_model, "phi3-mini-idf-lora")
model = model.merge_and_unload()

# Generate IDF
prompt = "<|user|>\nCreate a Site:Location for Tokyo with latitude 35.68<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.1,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
