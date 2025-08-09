import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    base_model_name = "microsoft/phi-2"
    adapter_path = "models/phi2-eli5-adapter-r32-fresh/final_model"
    output_dir = "models/phi2-eli5-adapter-r32-fresh/merged_model"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        use_cache=False,
    )

    tokenizer=AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading LoRA adapter from: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model = model.merge_and_unload()  # Permenantly merge adapter wts into the base model and removes adapters from the memory\

    # --- Clean the state dictionary ---
    print("Cleaning model state dictionary...")
    state_dict = model.state_dict()
    keys_to_delete = [key for key in state_dict if key.endswith(".absmax")]
    for key in keys_to_delete:
        del state_dict[key]
        print(f"  - Removed tensor: {key}")
    print(f"Removed {len(keys_to_delete)} unnecessary tensors.")

    model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved successfully to :", {output_dir})

if __name__ == "__main__":
    main()