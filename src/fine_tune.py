#src/fine_tune.py

import os 
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

local_wandb_tmp_dir = os.path.join(os.getcwd(), "wandb_tmp")
os.makedirs(local_wandb_tmp_dir, exist_ok=True)

os.environ["WANDB_TEMP"] = local_wandb_tmp_dir
os.environ["WANDB_DIR"] = local_wandb_tmp_dir
os.environ["WANDB_CONFIG_DIR"] = local_wandb_tmp_dir


from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig
from trl import SFTTrainer
import config
import wandb

class LogPredictionsCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=10):
        self.tokenizer=tokenizer
        self.eval_dataset=eval_dataset.select(range(num_samples))
        self.num_samples=num_samples

    def on_evaluate(self, args, state, control, model, **kwargs):
        prompts = [config.formatting_func(ex) for ex in self.eval_dataset]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.SFT_MAX_SEQ_LENGTH
        ).to(model.device)
    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150
            )
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        questions = [str(ex['question']) for ex in self.eval_dataset]
        ground_truth_answers = [str(ex['answer']) for ex in self.eval_dataset]
    
        generated_answer = []
        answers_marker = "### Answer:\n"
    
        for pred in decoded_outputs:
            if answers_marker in pred:
                generated_answer.append(pred.split(answers_marker, 1)[1].strip())
            else:
                generated_answer.append("No answer found/Formatting error in generation")
    
        predictions_table = wandb.Table(columns=["Question", "Ground Truth", "Generated Answer"])
    
        for q, gt, gen in zip(questions, ground_truth_answers, generated_answer):
            predictions_table.add_data(str(q), str(gt), str(gen))  # Ensure all are strings
    
        wandb.log({"Evaluation Predictions": predictions_table})


def preprocess_function(examples, tokenizer):
    texts = [
        f"### Instruction:\nExplain the following like I'm 5: {q}\n\n### Answer:\n{a}"
        for q, a in zip(examples['question'], examples['answer'])
    ]

    return tokenizer(texts, truncation=True, max_length=config.SFT_MAX_SEQ_LENGTH)

def main():
    wandb.init(
        entity="BrainLoop",        
        project="explain-it-phi",
        name=f"phi2-run-{wandb.util.generate_id()}",  
        config=config.hyperparameters 
    )
    
    print("Loading and preprocessing datasets...")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = load_dataset("json", data_files=config.TRAIN_DATASET, split="train")
    val_dataset = load_dataset("json", data_files=config.VAL_DATASET, split="train")

    # Check if we are in testing mode to use a subset of the data
    if config.NUM_SAMPLES_FOR_TESTING:
        print(f"--- Running in TEST MODE: Using a subset of {config.NUM_SAMPLES_FOR_TESTING} samples. ---")
        train_dataset = train_dataset.select(range(config.NUM_SAMPLES_FOR_TESTING))
        val_sample_size = max(1, int(config.NUM_SAMPLES_FOR_TESTING * 0.2))
        val_dataset = val_dataset.select(range(val_sample_size))

    train_dataset = train_dataset.map(lambda exs: preprocess_function(exs, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda exs: preprocess_function(exs, tokenizer), batched=True)

    quant_config=BitsAndBytesConfig(**config.BNB_CONFIG)

    model=AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=quant_config,
        trust_remote_code=True,
        use_cache=False,
        device_map="auto",
    )

    peft_config=LoraConfig(**config.PEFT_CONFIG)

    training_arguments=TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        **config.TRAINING_ARGS,
    )

    prediction_callback = LogPredictionsCallback(
        tokenizer=tokenizer,
        eval_dataset=val_dataset
    )
    
    print("Initializing SFT Trainer.....")
    trainer=SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        args=training_arguments,
        callbacks=[prediction_callback],
    )
    print("W&B directories:")
    print("  WANDB_TEMP:", os.environ.get("WANDB_TEMP"))
    print("  WANDB_DIR:", os.environ.get("WANDB_DIR"))
    print("  WANDB_CONFIG_DIR:", os.environ.get("WANDB_CONFIG_DIR"))


    print("Starting Training.....")
    trainer.train()

    print("Saving fine tuned model adapters.....")
    trainer.save_model(config.OUTPUT_DIR)
    
    wandb.finish()
    
    print("Training Complete & Model Saved!")

if __name__=="__main__":
    main()