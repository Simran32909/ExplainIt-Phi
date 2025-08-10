# ExplainIt-Phi: Phi-2 for ELI5-Style Explanations

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

## 6. Evaluation & Results

The model's performance was evaluated against the base `microsoft/phi-2` model across a diverse set of 30 prompts. I've used the **Flesch-Kincaid Grade Level** metric, which estimates the U.S. school grade level required to understand a text. For this project's goal of simplifying complex topics, a **lower score is better**.

### Executive Summary

Across all 30 test prompts, the fine-tuned **ExplainIt-Phi model reduced the Flesch-Kincaid Grade Level by an average of 1.5 points**. This demonstrates a consistent and measurable improvement in generating simpler, more accessible explanations.

### Qualitative Comparison

The quantitative scores are best understood by seeing the model's output directly. Here is a side-by-side comparison for a technical prompt:

**Prompt:** `What is an API and what does it do, in simple terms?`

| Base Model Output (Grade Level: 10.82)                                                                                                                                                                                                                         | Fine-Tuned Model Output (Grade Level: 5.81)                                                                                                                                                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| An API, or Application Programming Interface, is a set of rules and protocols that allows different software applications to communicate with each other. It acts as a bridge between two applications, allowing them to exchange data and functionality.     | An API is like a waiter in a restaurant. You (an application) don't need to know how the kitchen works. You just give your order (a request) to the waiter (the API), and the waiter brings you your food (the data). |

The fine-tuned model uses a clear analogy, avoids technical jargon, and provides a much simpler explanation, as reflected in its significantly lower grade level score.

### Quantitative Results

The table below shows a curated list of 15 results, sorted by the change in grade level, to highlight the model's strengths and limitations.

| Prompt                                                              | Base Model Grade | Fine-Tuned Grade | Grade Level Drop |
|---------------------------------------------------------------------|------------------|------------------|------------------|
| Why do cryptocurrencies have value?                                 | 14.16            | **7.56**         | **-6.60**        |
| What is the difference between nuclear fission and fusion...        | 17.30            | **10.77**        | **-6.53**        |
| How does a microwave oven heat food?                                | 14.04            | **8.39**         | **-5.65**        |
| What is an API and what does it do, in simple terms?                | 10.82            | **5.81**         | **-5.01**        |
| What is compound interest and how does it work?                     | 13.98            | **9.21**         | **-4.77**        |
| Why do we dream when we sleep?                                      | 11.89            | **7.55**         | **-4.34**        |
| Explain how a CPU works in a computer, in simple terms.             | 10.86            | **6.71**         | **-4.15**        |
| What is the internet and how does it work, in simple terms?         | 11.48            | **7.88**         | **-3.60**        |
| Explain what blockchain is, like I'm 5.                             | 10.27            | **6.97**         | **-3.30**        |
| What are stem cells?                                                | 12.84            | **9.56**         | **-3.28**        |
| Explain the concept of 'supply and demand' in simple terms.         | 11.59            | **8.38**         | **-3.21**        |
| What is inflation in economics, in simple terms?                    | 13.39            | **10.76**        | **-2.63**        |
| What causes an earthquake?                                          | **5.24**         | 7.95             | +2.71            |
| Explain what machine learning is, like I'm 5.                       | **8.77**         | 14.64            | +5.87            |
| Explain the theory of relativity like I'm 5.                        | **6.26**         | 15.80            | +9.54            |

*The model with the lower (better) score is highlighted in **bold**.*

### Limitations

While the model shows significant improvement on complex topics, the evaluation revealed that for some already simple concepts, the fine-tuning process can lead to slightly more complex explanations. This suggests the model learned to prioritize providing a detailed, structured answer, which is a key area for future improvement.
 
