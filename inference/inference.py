import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    """
    Loads the base model and the fine-tuned LoRA adapter to run inference.
    """
    # Load the base model with quantization
    base_model_name = "microsoft/phi-2"
    
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

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and merge the LoRA adapter 
    adapter_path = "models/phi2-eli5-adapter-r32-fresh/final_model"
    print(f"Loading LoRA adapter from: {adapter_path}")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() # Merge weights and free memory

    # Generate text 
    prompt = "Why is trump raising tariffs?"
    input_text = f"Instruct: {prompt}\nOutput:"

    print(f"\nGenerating response for prompt: '{prompt}'")
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to("cuda")

    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("-" * 30)
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    main()