import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    base_model_name = "microsoft/phi-2"
    print("Loading base model : ", base_model_name)

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

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt = "Explain the concept of transformer architecture in deep learning in simple terms."
    input_text = f"Instruct: {prompt}\nOutput:"

    outputs= base_model.generate(
        **tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_length=256
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("-" * 30)
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    main()