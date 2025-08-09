import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """
    Loads the base model in FP16, merges the LoRA adapter, and saves the
    resulting full-precision model for GGUF conversion.
    """
    base_model_name = "microsoft/phi-2"
    adapter_path = "models/phi2-eli5-adapter-r32-fresh/final_model"
    # Save to a new directory to avoid confusion
    output_dir = "models/phi2-eli5-adapter-r32-fresh/merged_model_fp16"

    print("--- Loading base model in FP16 (this may take a moment and use more VRAM) ---")
    # IMPORTANT: Load the base model in float16, not 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"--- Loading LoRA adapter from: {adapter_path} ---")
    # Apply the adapter to the full-precision model
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    print("--- Merging adapter weights ---")
    # Merge the adapter into the model
    model = model.merge_and_unload()

    print(f"--- Saving merged FP16 model to: {output_dir} ---")
    # Save the full-precision merged model. This will be larger.
    model.save_pretrained(output_dir)

    # Save the tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)

    print(f"FP16 merged model saved successfully to {output_dir}")
    print("You can now use this directory for GGUF conversion.")

if __name__ == "__main__":
    main()