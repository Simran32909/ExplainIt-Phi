# src/lightning_module.py

import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig
from omegaconf import DictConfig, OmegaConf
import bitsandbytes as bnb
import math
from pathlib import Path
import os

def count_lines_in_file(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

class LLMLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters() # Saves cfg to the checkpoint

        # Convert string from YAML to torch.dtype
        compute_dtype = getattr(torch, self.cfg.model.bnb.bnb_4bit_compute_dtype)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.model.bnb.load_in_4bit,
            bnb_4bit_quant_type=self.cfg.model.bnb.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.cfg.model.bnb.bnb_4bit_use_double_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.base_model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
        
        # Add LoRA adapters
        peft_params = OmegaConf.to_container(
            self.cfg.model.peft, resolve=True
        )
        peft_config = LoraConfig(**peft_params)
        self.model.add_adapter(peft_config)
        self.model.gradient_checkpointing_enable()


    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # Use 8-bit Adam for memory efficiency
        optimizer = bnb.optim.PagedAdamW8bit(
            self.parameters(), 
            lr=self.cfg.optimizer.lr, 
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        train_file_path = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ) / self.cfg.data.train_file

        # Calculate dataset size
        if hasattr(self.cfg.data, 'num_samples_for_testing') and self.cfg.data.num_samples_for_testing:
            num_samples = self.cfg.data.num_samples_for_testing
            print(f"Using TESTING subset: {num_samples} samples")
        else:
            num_samples = count_lines_in_file(train_file_path)
            print(f"Using FULL dataset: {num_samples} samples")

        steps_per_epoch = num_samples // (self.cfg.data.batch_size * self.cfg.trainer.accumulate_grad_batches)
        total_steps = steps_per_epoch * self.cfg.trainer.max_epochs
        warmup_steps = int(total_steps * self.cfg.scheduler.warmup_ratio)

        print(
            f"Scheduler steps - Total: {total_steps}, Warmup: {warmup_steps}, "
            f"Batch size: {self.cfg.data.batch_size}, Grad accumulation: {self.cfg.trainer.accumulate_grad_batches}"
        )

        if self.cfg.scheduler.type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.cfg.scheduler.type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.cfg.scheduler.num_cycles
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.cfg.scheduler.type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }