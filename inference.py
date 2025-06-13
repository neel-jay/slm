# inference.py

import torch
from model import ReasoningModel

# This is the path where your trained model was saved.
MODEL_PATH = "./reasoning_model_finetuned"

def run_inference():
    """
    Loads the fine-tuned model and starts an interactive session.
    """
    try:
        # Initialize the model class and load the fine-tuned weights
        reasoning_model = ReasoningModel()
        reasoning_model.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{MODEL_PATH}'.")
        print("Please run the 'train.py' script first to train and save the model.")
        return

    tokenizer = reasoning_model.tokenizer
    model = reasoning_model.model
    device = reasoning_model.device

    print("\n--- Chain-of-Thought Reasoning Model Ready ---")
    print("Type your question, or type 'exit' to quit.")

    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break

        # Prepare the input for the model
        input_text = f"question: {question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate the response
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=256,
                num_beams=5, # Use beam search for higher quality output
                early_stopping=True
            )

        raw_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # --- Extract only the final answer for a cleaner display ---
        if "Final Answer:" in raw_response:
            final_answer = raw_response.split("Final Answer:")[-1].strip()
        else:
            final_answer = raw_response # Fallback if the model doesn't use the format

        print("AI:", final_answer)
        # Optional: To see the model's "thought" process, uncomment the line below
        # print("\n--- Full Generation (with thought process) ---\n", raw_response)


if __name__ == "__main__":
    run_inference()