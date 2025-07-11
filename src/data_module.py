# src/data_module.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from omegaconf import DictConfig
import torch

def formatting_func(example):
    text = f"### Instruction:\nExplain the following like I'm 5: {example['question']}\n\n### Answer:\n{example['answer']}"
    return text

class LLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Left padding for more efficient training

    def prepare_data(self):
        load_dataset("json", data_files=self.cfg.data.train_file)
        load_dataset("json", data_files=self.cfg.data.val_file)

    def setup(self, stage=None):
        train_dataset = load_dataset("json", data_files=self.cfg.data.train_file, split="train")
        val_dataset = load_dataset("json", data_files=self.cfg.data.val_file, split="train")

        if self.cfg.data.num_samples_for_testing:
            num_train_samples = min(self.cfg.data.num_samples_for_testing, len(train_dataset))
            train_dataset = train_dataset.select(range(num_train_samples))

            val_sample_size = max(1, int(num_train_samples * 0.1))  # Reduced validation size
            num_val_samples = min(val_sample_size, len(val_dataset))
            val_dataset = val_dataset.select(range(num_val_samples))

        # We define a nested function here to capture self.tokenizer and self.cfg
        def preprocess_function(examples):
            texts = [
                f"### Instruction:\nExplain the following like I'm 5: {q}\n\n### Answer:\n{a}"
                for q, a in zip(examples['question'], examples['answer'])
            ]
            
            # Tokenize with dynamic padding for training efficiency
            model_inputs = self.tokenizer(
                texts, 
                truncation=True, 
                max_length=self.cfg.model.sft_max_seq_length, 
                padding='max_length',
                return_tensors="pt"
            )
            
            # Set labels for causal language modeling
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            
            # Create attention mask that ignores padding tokens
            attention_mask = model_inputs["attention_mask"]
            
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": attention_mask,
                "labels": model_inputs["labels"]
            }

        # Use batched processing without multiprocessing to avoid CUDA issues
        self.train_dataset = train_dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=train_dataset.column_names,
            batch_size=100  # Process in larger batches
        )
        
        self.val_dataset = val_dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=val_dataset.column_names,
            batch_size=100
        )

        # Set format to torch for the dataloaders
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            shuffle=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False
        ) 