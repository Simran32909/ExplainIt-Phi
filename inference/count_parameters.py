import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

def count_safetensor_parameters(file_path):
    """
    Counts the total number of parameters in a .safetensors file.
    """
    try:
        tensors = load_file(file_path)
        total_params = 0
        for tensor_name in tensors:
            tensor = tensors[tensor_name]
            total_params += tensor.numel()
        return total_params
    except Exception as e:
        print(f"An error occurred while reading safetensors file: {e}")
        return None

def count_hf_model_parameters(model):
    """
    Counts the total number of parameters in a Hugging Face model.
    """
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # --- 1. Count fine-tuned parameters from adapter ---
    adapter_model_path = "/ssd_scratch/jyothi.swaroopa/Simran/ExplainIt-Phi/models/phi2-eli5-adapter-r32-fresh/final_model/adapter_model.safetensors"
    ft_params = count_safetensor_parameters(adapter_model_path)
    if ft_params is not None:
        print(f"Total number of fine-tuned parameters in adapter: {ft_params:,}")

    # --- 2. Count total parameters in the base model ---
    base_model_name = "microsoft/phi-2"
    print(f"\nLoading base model ({base_model_name}) to count parameters...")
    
    try:
        # Load the model on CPU without quantization for accurate counting
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        base_params = count_hf_model_parameters(base_model)
        print(f"Total number of parameters in base model: {base_params:,}")

        if ft_params is not None:
            trainable_percentage = (ft_params / base_params) * 100
            print(f"Fine-tuned parameters are {trainable_percentage:.4f}% of the base model.")

    except Exception as e:
        print(f"An error occurred while loading the base model: {e}")
