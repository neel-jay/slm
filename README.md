# Project Title: Chain-of-Thought Reasoning Model (with T5 and LM Studio/DeepSeek Integration)

## Overview
This project implements a Chain-of-Thought (CoT) reasoning model. It originally focused on fine-tuning and using a T5 transformer model. It has been extended to integrate with local LLMs, such as DeepSeek, running via LM Studio.

The project now allows users to:
1.  Fine-tune a pre-trained T5 model on custom CoT data (original functionality).
2.  Interact with the fine-tuned T5 model (original functionality).
3.  **NEW:** Use a locally running model (e.g., DeepSeek via LM Studio) to generate new CoT training examples.
4.  **NEW:** Interact with a locally running model (e.g., DeepSeek via LM Studio) for CoT inference.

## How it Works

### Original T5-based Workflow
-   **T5 Model (`google/flan-t5-small`):** A pre-trained language model used as the base for fine-tuning.
-   **Chain-of-Thought Fine-tuning:** The T5 model can be fine-tuned on examples where the answer includes a step-by-step thought process.
-   **Key Scripts (T5):**
    -   `model.py`: Defines the `ReasoningModel` class for T5.
    -   `training_data.py`: Provides the *initial* CoT training examples for T5.
    -   `train.py`: Script to fine-tune the T5 model.
    -   `inference.py`: Script to interact with the fine-tuned T5 model.

### NEW: LM Studio Integration (e.g., with DeepSeek)
This integration allows leveraging powerful local LLMs managed by [LM Studio](https://lmstudio.ai/).
-   **LM Studio:** A desktop application for discovering, downloading, and running local LLMs. It provides a server to interact with these models.
-   **LM Studio Python SDK (`lmstudio`):** Used by this project to communicate with the LM Studio server.
-   **Key Scripts (LM Studio):**
    -   `generate_new_training_data.py`: Uses a model running in LM Studio (e.g., DeepSeek) to generate new CoT question-answer pairs. These can potentially be used to augment training data for other models (like T5) or for analysis.
    -   `inference_deepseek.py`: Allows direct CoT inference using a model (e.g., DeepSeek) running in LM Studio.

## Setup and Installation

### Prerequisites
- Python 3.7+
- Pip (Python package installer)
- **For LM Studio Integration:**
    - [LM Studio](https://lmstudio.ai/) application installed and running.
    - A model (e.g., DeepSeek `deepseek/deepseek-r1-0528-qwen3-8b`) downloaded and loaded within LM Studio.
    - The LM Studio server must be started (usually on `localhost:1234`).

### Dependencies
This project relies on several Python libraries. A `requirements.txt` file is provided.

**To install all dependencies:**
```bash
pip install -r requirements.txt
```

This includes:
- `torch`: For PyTorch functionalities (used by the T5 part).
- `transformers`: From Hugging Face, for the T5 model and tokenizer.
- `sentencepiece`: Often required by T5 tokenizers.
- `tqdm`: Used for progress bars during T5 training.
- `lmstudio`: The official Python SDK for LM Studio.

If you only intend to use one part of the project (e.g., only LM Studio integration or only T5), you might be able to install selectively, but the `requirements.txt` includes all.

## Using the LM Studio Integration

### 1. Setting up LM Studio
1.  Download and install [LM Studio](https://lmstudio.ai/).
2.  Launch LM Studio. Search for and download your desired model (e.g., `deepseek/deepseek-r1-0528-qwen3-8b`).
3.  Load the model.
4.  Navigate to the "Local Server" tab (often an icon of a computer or network).
5.  Click "Start Server".

### 2. Generating New CoT Data with DeepSeek (via LM Studio)
The `generate_new_training_data.py` script uses a model in LM Studio to create new question-answer pairs in the CoT format.

**To run:**
```bash
python generate_new_training_data.py
```
- The script will attempt to connect to the model specified by `MODEL_IDENTIFIER` (default: `deepseek/deepseek-r1-0528-qwen3-8b`). You can change this in the script or via the `LMSTUDIO_MODEL_IDENTIFIER` environment variable.
- Generated data will be saved to `generated_training_data.json`.
- **Note:** This script requires your LM Studio server to be running with the specified model loaded.

### 3. Performing Inference with DeepSeek (via LM Studio)
The `inference_deepseek.py` script allows you to chat with your local model for CoT-style responses.

**To run:**
```bash
python inference_deepseek.py
```
- It connects to the model specified by `MODEL_IDENTIFIER` (default: `deepseek/deepseek-r1-0528-qwen3-8b`).
- You'll be prompted to enter questions in the console.
- Type `exit` or `quit` to end the session.
- **Note:** This script also requires your LM Studio server to be running with the specified model loaded.

## Original T5 Model Workflow

### Training the T5 Model
To fine-tune the T5 model with the data in `training_data.py` (or your custom CoT data added there):
```bash
python train.py
```
**Configuration:** See `EPOCHS`, `BATCH_SIZE`, etc., at the top of `train.py`.
The fine-tuned model is saved to `./reasoning_model_finetuned`.

### Running Inference with the Fine-tuned T5 Model
To interact with your fine-tuned T5 model:
```bash
python inference.py
```
This loads the model from `./reasoning_model_finetuned`.

## Customizing Training Data

### For T5 Fine-tuning:
Edit `training_data.py`. The `get_cot_training_data()` function returns a list of dictionaries:
```python
{
    "question": "Your question here",
    "answer": "Thought: Your reasoning steps. Final Answer: Your final answer."
}
```

### For LM Studio-based Generation:
The meta-prompt within `generate_new_training_data.py` (in the `generate_cot_example` function) instructs the LLM on how to generate data. You can modify this meta-prompt to change the style or content of the generated examples. The generated examples are saved in `generated_training_data.json`.
