#src/config.py

import torch

BASE_MODEL_NAME="microsoft/phi-2"

NUM_SAMPLES_FOR_TESTING=None

TRAIN_DATASET="data/processed/train.json"
VAL_DATASET="data/processed/validation.json"

#saves the fine tuned model
OUTPUT_DIR="models/phi2-eli5-adapter"

SFT_MAX_SEQ_LENGTH=1024

#Quantizations Configs
BNB_CONFIG={
    "load_in_4bit":True,
    "bnb_4bit_quant_type":"nf4",
    "bnb_4bit_compute_dtype":torch.float16,  # Changed from "bfloat16" to torch.float16
    "bnb_4bit_use_double_quant":True,       # Added for better quantization
}

#LORA Configs
PEFT_CONFIG={
    "r":32,
    "lora_alpha":64,
    "lora_dropout":0.05,
    "bias":"none",
    "task_type":"CAUSAL_LM",
    "target_modules": ["Wqkv", "fc1", "fc2", "q_proj", "k_proj", "v_proj"],
}

#Training Arguements
TRAINING_ARGS={
    "num_train_epochs":3,
    "per_device_train_batch_size":6,
    "per_device_eval_batch_size":6,
    "gradient_accumulation_steps":3,
    "optim":"paged_adamw_8bit",
    "save_steps": 100,
    "logging_steps": 50,
    "learning_rate": 3e-5,
    "weight_decay": 0.002,
    "fp16": True,        # Changed from False to True
    "bf16": False,       # Changed from True to False
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "eval_strategy": "steps",
    "dataloader_pin_memory": True,    
    "dataloader_num_workers": 4,
    "eval_steps": 100,
    "report_to": "wandb",
    "torch_compile": False,
    "save_total_limit": 3,
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