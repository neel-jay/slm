# model.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

class ReasoningModel:
    """
    A class that encapsulates the T5 model and tokenizer for reasoning tasks.
    It handles loading the model from Hugging Face or from a local checkpoint.
    """
    def __init__(self, model_name='google/flan-t5-small'):
        """
        Initializes the model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model to load.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        print(f"Model '{self.model_name}' loaded on {self.device}.")

    def save_model(self, save_path):
        """
        Saves the fine-tuned model and tokenizer to the specified path.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Loads a fine-tuned model and tokenizer from a local path.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model path not found: {load_path}")
        
        self.model = T5ForConditionalGeneration.from_pretrained(load_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        print(f"Model loaded from {load_path}")