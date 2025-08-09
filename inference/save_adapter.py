
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path

from lightning_module import LLMLightningModule

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = "/ssd_scratch/jyothi.swaroopa/Simran/ExplainIt-Phi/models/phi2-eli5-adapter-r32-fresh/explain-it-phi-epoch=02-val_loss=0.58-val_perplexity=1.80.ckpt"
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model_module = LLMLightningModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,  # Pass the config to the model
        strict=False  # Ignore non-matching keys
    )
    output_dir = Path(cfg.output_dir)
    final_model_path = output_dir / "final_model"
    os.makedirs(final_model_path, exist_ok=True)
    print(f"Saving LoRA adapter to: {final_model_path}")
    model_module.model.save_pretrained(final_model_path)
    print("Adapter saved successfully!")
    print(f"Files in final_model directory: {os.listdir(final_model_path)}")

if __name__ == "__main__":
    main()
