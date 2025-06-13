# Project Title: Chain-of-Thought Reasoning Model

## Overview
This project implements a Chain-of-Thought (CoT) reasoning model using a fine-tuned T5 transformer. It allows users to:
1. Fine-tune a pre-trained T5 model on custom question-answering data that includes reasoning steps.
2. Interact with the fine-tuned model for question answering, observing the model's thought process.

## How it Works
The project is built around the following core components:
- **T5 Model (`google/flan-t5-small`):** A powerful pre-trained language model from Google, used as the base for fine-tuning.
- **Chain-of-Thought Fine-tuning:** The model is fine-tuned on examples where the answer includes a step-by-step thought process, enabling the model to generate similar reasoning for new questions.
- **Key Scripts:**
    - `model.py`: Defines the `ReasoningModel` class, which handles loading, saving, and managing the T5 model and tokenizer.
    - `training_data.py`: Provides the training examples for CoT fine-tuning. You can customize this data.
    - `train.py`: Contains the script to fine-tune the T5 model using the data from `training_data.py`.
    - `inference.py`: Allows users to load the fine-tuned model and interact with it for question answering.

## Setup and Installation

### Prerequisites
- Python 3.7+
- Pip (Python package installer)

### Dependencies
This project relies on several Python libraries. You can install them using pip:
```bash
pip install torch transformers sentencepiece tqdm
```
(Note: `sentencepiece` is often required by T5 tokenizers. `tqdm` is used for progress bars during training.)

It is also recommended to create a `requirements.txt` file for easier dependency management. See the "Dependencies" section below.

## Training the Model

To fine-tune the model with the provided chain-of-thought data, run the `train.py` script:

```bash
python train.py
```

### Training Configuration
The training script (`train.py`) has several key configuration variables at the top of the file that you can modify:

- `MODEL_SAVE_PATH = "./reasoning_model_finetuned"`: The directory where the fine-tuned model and tokenizer will be saved.
- `EPOCHS = 10`: The number of times the training process will iterate over the entire dataset.
- `BATCH_SIZE = 2`: The number of training examples processed before the model's weights are updated. You might need to adjust this based on your GPU memory.
- `LEARNING_RATE = 5e-5`: The learning rate for the optimizer.

The script will:
1. Load the base `google/flan-t5-small` model.
2. Load the training data from `training_data.py`.
3. Tokenize the data.
4. Fine-tune the model using the specified configurations.
5. Save the fine-tuned model and tokenizer to the `MODEL_SAVE_PATH`.

Upon completion, you will see a message indicating that training is finished and the model is ready for inference.

## Running Inference

Once the model has been trained and saved (by running `train.py`), you can interact with it using the `inference.py` script:

```bash
python inference.py
```

This script will:
1. Load the fine-tuned model from the path specified by `MODEL_PATH` (default is `./reasoning_model_finetuned`) in `inference.py`.
2. Start an interactive command-line session.
3. Prompt you to enter your questions.

Type your question and press Enter. The AI will then generate a response, including its chain-of-thought reasoning (if the model produces it) and the final answer.

To exit the interactive session, type `exit`.

If the script cannot find the trained model, it will print an error message prompting you to run `train.py` first.

## Customizing Training Data

The quality and nature of the fine-tuning data significantly impact the model's performance and its ability to perform chain-of-thought reasoning. You can customize the training data by editing the `training_data.py` file.

The `get_cot_training_data()` function in this file returns a list of dictionaries. Each dictionary represents a single training example and must have two keys:
- `"question"`: A string containing the input question.
- `"answer"`: A string containing the desired output, which should include the "Thought:" process leading to the "Final Answer:".

**Example format:**
```python
{
    "question": "I have 5 apples and I buy 3 more boxes of apples. Each box contains 12 apples. How many apples do I have in total?",
    "answer": "Thought: The user starts with 5 apples. They buy 3 more boxes. Each box has 12 apples. First, I need to calculate the total number of new apples from the boxes. That is 3 boxes * 12 apples/box = 36 apples. Then, I need to add this to the initial number of apples. So, 36 new apples + 5 initial apples = 41 apples.\nFinal Answer: You have 41 apples in total."
}
```

To use your custom data:
1. Open `training_data.py`.
2. Modify the list returned by `get_cot_training_data()` with your own question/answer pairs, following the format above.
3. Ensure your answers clearly demonstrate a step-by-step reasoning process before stating the final answer.
4. After customizing the data, retrain the model by running `python train.py`.

The more high-quality and relevant examples you provide, the better the model will adapt to your specific task or domain.

## Dependencies

This project relies on the following Python libraries:
- `torch`: For PyTorch functionalities.
- `transformers`: From Hugging Face, for the T5 model and tokenizer.
- `sentencepiece`: Often required by T5 tokenizers.
- `tqdm`: Used for progress bars during training.

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```
This `requirements.txt` file is included in the repository.

Alternatively, you can install them individually as mentioned in the "Setup and Installation" section:
```bash
pip install torch transformers sentencepiece tqdm
```
