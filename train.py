# train.py

import torch
# Corrected import: AdamW is now imported from torch.optim
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader, TensorDataset
from model import ReasoningModel
from training_data import get_cot_training_data
from tqdm import tqdm # For a nice progress bar

# --- Configuration ---
MODEL_SAVE_PATH = "./reasoning_model_finetuned"
EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 5e-5

def train_model():
    """
    Main function to handle the model training process.
    """
    # 1. Initialize the model
    reasoning_model = ReasoningModel(model_name='google/flan-t5-small')
    tokenizer = reasoning_model.tokenizer

    # 2. Load and prepare the training data
    training_data = get_cot_training_data()
    
    inputs = [f"question: {item['question']}" for item in training_data]
    targets = [item['answer'] for item in training_data]

    # Tokenize all data
    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

    # The labels are the target token ids
    labels = target_encodings.input_ids
    # T5 requires padding tokens in labels to be replaced by -100
    labels[labels == tokenizer.pad_token_id] = -100

    # Create a PyTorch dataset
    dataset = TensorDataset(
        input_encodings.input_ids,
        input_encodings.attention_mask,
        labels
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Set up optimizer and scheduler
    optimizer = AdamW(reasoning_model.model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # 4. Run the training loop
    print("--- Starting Fine-Tuning ---")
    reasoning_model.model.train() # Set model to training mode
    
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0

        for batch in progress_bar:
            # Move batch to the correct device
            batch = [t.to(reasoning_model.device) for t in batch]
            input_ids, attention_mask, batch_labels = batch
            
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = reasoning_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            
            # Calculate loss and perform backpropagation
            loss = outputs.loss
            loss.backward()
            
            # Update weights
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    print("--- Fine-Tuning Complete ---")
    
    # 5. Save the fine-tuned model
    reasoning_model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()
    print(f"\nTraining finished. You can now use inference.py to interact with the model located at '{MODEL_SAVE_PATH}'.")