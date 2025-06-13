import os
import json
from lmstudio.client import LMSClient
from lmstudio.exceptions import LMSException

# Model identifier - user can change this if needed
MODEL_IDENTIFIER = os.getenv("LMSTUDIO_MODEL_IDENTIFIER", "deepseek/deepseek-r1-0528-qwen3-8b")
OUTPUT_FILE = "generated_training_data.json"

def get_lmstudio_client():
    """Initializes and returns an LMSClient instance."""
    try:
        client = LMSClient() # Using default connection params (localhost:1234)
        print("Successfully created LMSClient.")
        return client
    except LMSException as e:
        print(f"Error creating LMSClient: {e}")
        print("Please ensure LM Studio is running and the server is enabled.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while creating LMSClient: {e}")
        return None

def get_model_reference(client, model_identifier):
    """Gets a reference to the specified model using the client."""
    if not client:
        return None
    try:
        print(f"Attempting to get model reference for: {model_identifier}")
        # List loaded models for debugging help
        try:
            loaded_models = client.list_loaded_models()
            if loaded_models:
                print("\n--- Currently Loaded Models in LM Studio ---")
                for i, model_info in enumerate(loaded_models):
                    print(f"{i+1}. Name: {model_info.name}, Path: {model_info.path}, ID: {model_info.id}")
                    if model_identifier.lower() in model_info.path.lower() or \
                       model_identifier.lower() in model_info.name.lower():
                        print(f"   ^-- This appears to be a match for '{model_identifier}'")
                print("-------------------------------------------\n")
            else:
                print("No models reported as loaded by the SDK. Ensure your model is loaded in LM Studio.")
        except Exception as e:
            print(f"Could not list loaded models: {e}")

        model = client.llm(model_identifier)
        print(f"Successfully got model reference for: {model_identifier}")
        return model
    except LMSException as e:
        print(f"Error getting model reference for '{model_identifier}': {e}")
        print("Please ensure the model identifier is correct and the model is loaded/available in LM Studio.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting model reference: {e}")
        return None

def generate_cot_example(model, topic):
    """
    Generates a Chain-of-Thought (CoT) question and answer pair on a given topic
    using the provided model.
    """
    if not model:
        print("Model reference not available. Skipping generation.")
        return None

    # This is the meta-prompt. It instructs DeepSeek on how to generate the data.
    meta_prompt = f"""
You are an expert data generator for training AI models.
Your task is to create a question and a detailed Chain-of-Thought (CoT) answer for the given topic.
The CoT answer must first show a step-by-step thought process (beginning with "Thought:") and then conclude with the final answer (beginning with "Final Answer:").

Topic: {topic}

Generate a unique question related to this topic and provide the CoT answer.
Example format:
Question: [Generated Question Here]
Answer: Thought: [Step-by-step reasoning here].
Final Answer: [Final concise answer here]

Do not include the "Topic:" line in your actual output. Start directly with "Question:".
"""

    print(f"Sending meta-prompt to DeepSeek for topic: {topic}")

    try:
        # Using model.respond for chat-like interaction.
        # We might need to adjust predict_kwargs for better control.
        # max_tokens should be sufficient for a detailed CoT answer.
        response_text = model.respond(meta_prompt, predict_kwargs={"temperature": 0.7, "max_tokens": 512})

        print(f"Raw response from DeepSeek:\n{response_text}")

        # Basic parsing of the response (this will likely need refinement)
        if isinstance(response_text, str):
            if "Question:" in response_text and "Answer:" in response_text:
                parts = response_text.split("Question:", 1)[1].split("Answer:", 1)
                question = parts[0].strip()
                answer = parts[1].strip()

                # Further check for CoT structure in answer
                if "Thought:" in answer and "Final Answer:" in answer:
                    return {"question": question, "answer": answer}
                else:
                    print("Warning: Generated answer does not strictly follow CoT format (Thought/Final Answer).")
                    # Still return it for now, might need manual check or prompt refinement
                    return {"question": question, "answer": answer, "format_warning": True}
            else:
                print("Warning: Could not parse 'Question:' and 'Answer:' from DeepSeek response.")
                return {"raw_response": response_text, "parsing_error": True} # Store raw if parsing fails
        else:
            print(f"Warning: Unexpected response type from model: {type(response_text)}")
            return {"raw_response": str(response_text), "type_error": True}

    except LMSException as e:
        print(f"Error during model.respond: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        return None

def save_data(data_list, filename):
    """Saves the list of generated data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f, indent=4)
        print(f"Successfully saved {len(data_list)} examples to {filename}")
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")

def main():
    print("--- Starting Data Generation Script ---")
    client = get_lmstudio_client()
    if not client:
        return

    # Use a context manager for the client if it supports it,
    # otherwise, we'll rely on it closing when the script ends.
    # The LMSClient from lmstudio.client.LMSClient is designed to be used with 'with'
    with client:
        model = get_model_reference(client, MODEL_IDENTIFIER)
        if not model:
            print("Could not obtain model reference. Exiting.")
            return

        generated_data = []

        # Example topics for generation
        topics = [
            "calculating the area of a composite shape",
            "the process of photosynthesis",
            "historical significance of the Silk Road"
        ]

        for topic in topics:
            print(f"\n--- Generating example for topic: {topic} ---")
            example = generate_cot_example(model, topic)
            if example:
                generated_data.append(example)
                print(f"Generated example: {example}")

        if generated_data:
            save_data(generated_data, OUTPUT_FILE)
        else:
            print("No data was generated.")

    print("--- Data Generation Script Finished ---")

if __name__ == "__main__":
    main()
