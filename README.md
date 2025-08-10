# ExplainIt-Phi: Fine-Tuned Phi-2 for "Explain-me-like-I'm-5"

This repo contains the code and workflow for fine-tuning the `microsoft/phi-2` model using the QLoRA. The final output is a set of GGUF models suitable for high-performance local inference with `llama.cpp`.

## Dataset

This model was fine-tuned on the [`sentence-transformers-eli5`](https://huggingface.co/datasets/sentence-transformers/eli5) dataset. The data was processed to follow an `Instruct: <question>\n Output: <answer>` format to teach the model how to respond to prompts as a helpful explainer.

## Setup

### Initialize Submodules
This project uses `llama.cpp` as a submodule.
```bash
git submodule update --init --recursive
```

## 1. Fine-Tuning

The fine-tuning process is orchestrated by Hydra. All configurations are located in `conf/config.yaml`.

To start the training process:
```bash
python src/fine_tune.py
```
This will produce PyTorch Lightning checkpoints in the output dir

## 2. Merging the QLoRA Adapter

Post training, the QLoRA adapter must be merged into the base model to create a standalone model for conversion.

```bash
python src/merge_adapters.py
```
#### What is happening?
- Loads the base `phi-2` model in FP16,
- Applies the trained QLoRA adapter from the best checkpoint, and saves the result to a new directory (e.g., `models/phi2-eli5-adapter-r32-fresh/merged_model_fp16`).

## 3. GGUF Conversion & Quantization

Uses the `llama.cpp` tools to convert the merged model into the GGUF format.

### 3.1 Build `llama.cpp`
First, compile the necessary C++ tools.
```bash
cd llama.cpp~
mkdir build
cd build
cmake ..
cmake --build .
cd ../..
```
*Note: Add flags to the `cmake ..` command, e.g., `-DGGML_CUDA=on` for NVIDIA GPU support.*

### 3.2 Convert to FP16 GGUF
Create a high-quality, unquantized GGUF file from the merged model.
```bash
python llama.cpp~/convert_hf_to_gguf.py \
  models/phi2-eli5-adapter-r32-fresh/merged_model_fp16 \
  --outfile "ExplainIt-Phi-F16.gguf" \
  --outtype f16
```

### 3.3 Quantize the Model
Create a smaller, faster, quantized version for inference. The Q4_K_M version is recommended.
```bash
./llama.cpp~/build/bin/llama-quantize \
  ExplainIt-Phi-F16.gguf \
  ExplainIt-Phi-Q4_K_M.gguf \
  Q4_K_M
```

## 4. Running Local Inference

Use the `llama-cli` tool from the `llama.cpp` build to run the final quantized model.

```bash
./llama.cpp~/build/bin/llama-cli \
  -m ExplainIt-Phi-Q4_K_M.gguf \
  -p "Instruct: Explain the concept of a Large Language Model in simple terms.\nOutput:" \
  -n 256 \
  -c 2048 \
  --color
```

The model expects prompts in the format: `Instruct: <your prompt>\nOutput:`.

## 5. Model Artifacts

The final, quantized GGUF models are available for download on the HuggingFace Hub.

- **Hugging Face Repository:** [**`simraann/ExplainIt-Phi-GGUF`**](https://huggingface.co/simraann/ExplainIt-Phi-GGUF) 
The following quantized models are provided:
- **`ExplainIt-Phi-Q4_K_M.gguf`**: The recommended version for a great balance of performance and quality.
- **`ExplainIt-Phi-Q5_K_M.gguf`**: A higher-quality version for systems with more memory.
- **`ExplainIt-Phi-Q8_0.gguf`**: A near-lossless version for maximum quality on capable hardware.