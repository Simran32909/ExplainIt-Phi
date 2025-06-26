# src/lightning_module.py

import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from omegaconf import DictConfig
import bitsandbytes as bnb
import math

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
        peft_config = LoraConfig(**self.cfg.model.peft)
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
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
        
        # Configure scheduler
        if self.cfg.scheduler.type == "linear":
            from transformers import get_linear_schedule_with_warmup
            
            # Calculate total steps
            train_dataloader = self.trainer.train_dataloader
            if isinstance(train_dataloader, list):
                train_dataloader = train_dataloader[0]
            
            total_steps = len(train_dataloader) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches
            warmup_steps = int(total_steps * self.cfg.scheduler.warmup_ratio)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        
        return optimizer 

