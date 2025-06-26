# src/fine_tune.py

import os

project_root = os.getcwd()  
cache_dir = os.path.join(project_root, "cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cache_dir

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy

from data_module import LLMDataModule
from lightning_module import LLMLightningModule

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up local W&B temporary directories
local_wandb_tmp_dir = os.path.join(os.getcwd(), "wandb_tmp")
os.makedirs(local_wandb_tmp_dir, exist_ok=True)
os.environ["WANDB_TEMP"] = local_wandb_tmp_dir
os.environ["WANDB_DIR"] = local_wandb_tmp_dir
os.environ["WANDB_CONFIG_DIR"] = local_wandb_tmp_dir


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.project_name}-run-{hydra.utils.get_original_cwd().split('/')[-1]}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Initialize DataModule
    data_module = LLMDataModule(cfg)

    # Initialize LightningModule
    model_module = LLMLightningModule(cfg)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output_dir,
        save_top_k=cfg.trainer.save_top_k,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.mode,
        filename=f"{cfg.project_name}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_perplexity:.2f}}",
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor=cfg.trainer.monitor,
        patience=cfg.trainer.early_stopping_patience,
        mode=cfg.trainer.mode,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=0.5,  # Check validation set twice per epoch
        gradient_clip_val=cfg.trainer.max_grad_norm,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        deterministic=False,  # Faster training
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=1,  # Only run one batch for sanity check
    )

    # Start training
    print("Starting training with PyTorch Lightning...")
    trainer.fit(model_module, datamodule=data_module)
    print("Training finished.")

    # Save the final model adapter
    final_model_path = os.path.join(cfg.output_dir, "final_model")
    model_module.model.save_pretrained(final_model_path)
    print(f"Final model adapter saved to {final_model_path}")
    
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()