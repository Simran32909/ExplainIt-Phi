import torch
import textstat
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "models/phi2-eli5-adapter-r32-fresh/final_model" # Update this path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    # --- Technology & Computer Science ---
    "Explain how a CPU works in a computer, in simple terms.",
    "What is the internet and how does it work, in simple terms?",
    "Explain what blockchain is, like I'm 5.",
    "How do GPS satellites know where you are, in simple terms?",
    "What is an API and what does it do, in simple terms?",
    "What is the difference between RAM and storage on a computer?",
    "Explain what machine learning is, like I'm 5.",
    "How does a search engine like Google find information?",

    # --- Physics & Cosmology ---
    "Explain the concept of black holes in simple terms.",
    "What is the difference between nuclear fission and fusion, in simple terms?",
    "Explain the theory of relativity like I'm 5.",
    "What is dark matter and why do scientists think it exists?",
    "How does a rainbow form?",

    # --- Biology & Health ---
    "How does photosynthesis work, in simple terms?",
    "How do vaccines work to protect us from diseases?",
    "What is DNA and what does it do, in simple terms?",
    "Explain the human immune system like I'm 5.",
    "What are stem cells?",

    # --- Economics & Finance ---
    "What is inflation in economics, in simple terms?",
    "Explain the concept of 'supply and demand' in simple terms.",
    "What is compound interest and how does it work?",
    "Why do cryptocurrencies have value?",

    # --- Earth Science & Everyday Phenomena ---
    "What causes the seasons, in simple terms?",
    "How does soap clean things, in simple terms?",
    "How does a microwave oven heat food?",
    "What causes an earthquake?",
    "Explain the water cycle.",
    "How is glass made from sand?",
    "Why do we dream when we sleep?",
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
    # Set pad token for robust generation
    tokenizer.pad_token = tokenizer.eos_token
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
    
    # Set pad token for robust generation
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    input_text = f"Instruct: {prompt}\nOutput:"
    # Let the tokenizer create the attention mask
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(DEVICE)
    
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
