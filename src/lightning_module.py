# src/lightning_module.py

import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from omegaconf import DictConfig
import bitsandbytes as bnb

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = bnb.optim.PagedAdamW8bit(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        return optimizer 

