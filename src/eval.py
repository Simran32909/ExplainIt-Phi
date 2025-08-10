import torch
import textstat
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "models/phi2-eli5-adapter-r32-fresh/final_model" # Update this path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "Explain the concept of black holes in simple terms.",
    "Why is the sky blue?",
    "Explain how a CPU works in a computer.",
    "What is the difference between nuclear fission and fusion?",
    "Explain the theory of relativity like I'm 5."
]

# --- Model Loading Functions (You will adapt your existing code here) ---

def load_base_model():
    print("Loading base Phi-2 model...")
    # Your code from inference_base.py to load the original phi-2 model
    # Return the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    return model, tokenizer

def load_finetuned_model():
    print("Loading fine-tuned ExplainIt-Phi model...")
    # Your code from merge_adapters.py or inference.py to load and merge
    # the adapter. Remember to load the base in FP16 for this.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    input_text = f"Instruct: {prompt}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False).to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200, # Generate a decent length response
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract just the generated part (the answer)
    response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response_full.split("Output:")[1].strip()
    return response_only

# --- Main Evaluation Logic ---

def main():
    base_model, base_tokenizer = load_base_model()
    finetuned_model, finetuned_tokenizer = load_finetuned_model()

    print("\n--- Readability Evaluation ---")
    print("-" * 50)
    
    results = []

    for prompt in PROMPTS:
        print(f"Processing prompt: '{prompt}'")
        
        # Generate from base model
        base_response = generate_response(base_model, base_tokenizer, prompt)
        base_grade = textstat.flesch_kincaid_grade(base_response)
        
        # Generate from fine-tuned model
        finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
        finetuned_grade = textstat.flesch_kincaid_grade(finetuned_response)
        
        results.append({
            "prompt": prompt,
            "base_grade": base_grade,
            "finetuned_grade": finetuned_grade
        })

    # Print results table
    print("\n--- Flesch-Kincaid Grade Level Results ---")
    print(f"{'Prompt':<55} | {'Base Model Grade':<20} | {'Fine-Tuned Grade':<20}")
    print("-" * 105)
    
    for res in results:
        print(f"{res['prompt']:<55} | {res['base_grade']:<20.2f} | {res['finetuned_grade']:<20.2f}")

if __name__ == "__main__":
    main()
