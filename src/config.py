#src/config.py

import torch

BASE_MODEL_NAME="microsoft/phi-2"

NUM_SAMPLES_FOR_TESTING=100

TRAIN_DATASET="data/processed/train.json"
VAL_DATASET="data/processed/validation.json"

#saves the fine tuned model
OUTPUT_DIR="models/phi2-eli5-adapter"

SFT_MAX_SEQ_LENGTH=1024

#Quantizations Configs
BNB_CONFIG={
    "load_in_4bit":True,
    "bnb_4bit_quant_type":"nf4",
    "bnb_4bit_compute_dtype":"float16",
}

#LORA Configs
PEFT_CONFIG={
    "r":16,
    "lora_alpha":32,
    "lora_dropout":0.08,
    "bias":"none",
    "task_type":"CAUSAL_LM",
    "target_modules": ["Wqkv", "fc1", "fc2"],
}

#Training Arguements
TRAINING_ARGS={
    "num_train_epochs":1,
    "per_device_train_batch_size":2,
    "gradient_accumulation_steps":4,
    "optim":"adamw_torch",
    "save_steps": 5,
    "logging_steps": 5,
    "learning_rate": 2e-3,
    "weight_decay": 0.002,
    "fp16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "constant",
    "eval_strategy": "steps",
    "eval_steps": 2,
    "report_to": "wandb",
}

def formatting_func(example):
    text = f"### Instruction:\nExplain the following like I'm 5: {example['question']}\n\n### Answer:\n{example['answer']}"
    return text

# clean dictionary for wandb logging
hyperparameters = {
    "base_model_name": BASE_MODEL_NAME,
    "train_dataset": TRAIN_DATASET,
    "val_dataset": VAL_DATASET,
    "output_dir": OUTPUT_DIR,
    "bnb_config": BNB_CONFIG,
    "peft_config": PEFT_CONFIG,
    "training_args": TRAINING_ARGS,
    "sft_max_seq_length": SFT_MAX_SEQ_LENGTH,
}