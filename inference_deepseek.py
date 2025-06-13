import os
from lmstudio.client import LMSClient
from lmstudio.exceptions import LMSException

# Model identifier - user can change this if needed, or set via environment variable
MODEL_IDENTIFIER = os.getenv("LMSTUDIO_MODEL_IDENTIFIER", "deepseek/deepseek-r1-0528-qwen3-8b")

def get_lmstudio_client():
    """Initializes and returns an LMSClient instance."""
    try:
        client = LMSClient() # Using default connection params (localhost:1234)
        print("Successfully created LMSClient for inference.")
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
        print(f"Attempting to get model reference for inference: {model_identifier}")
        # Optional: List loaded models for debugging help, similar to generation script
        # try:
        #     loaded_models = client.list_loaded_models()
        #     # ... (listing logic) ...
        # except Exception as e:
        #     print(f"Could not list loaded models: {e}")

        model = client.llm(model_identifier)
        print(f"Successfully got model reference for inference: {model_identifier}")
        return model
    except LMSException as e:
        print(f"Error getting model reference for '{model_identifier}': {e}")
        print("Please ensure the model identifier is correct and the model is loaded/available in LM Studio.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting model reference: {e}")
        return None

def run_inference_loop(model):
    """Runs the main interactive loop for inference with the DeepSeek model."""
    if not model:
        print("Model reference not available. Cannot start inference.")
        return

    print("\n--- DeepSeek CoT Inference Ready ---")
    print("Type your question, or type 'exit' or 'quit' to end.")

    while True:
        try:
            user_question = input("\nYou: ")
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting inference.")
                break

            if not user_question.strip():
                continue

            # Craft a prompt to encourage Chain-of-Thought reasoning
            # This prompt can be refined based on observed model behavior
            inference_prompt = f"""
Question: {user_question}

Provide a detailed step-by-step thought process (beginning with "Thought:") to arrive at the answer, and then state the final answer clearly (beginning with "Final Answer:").
"""

            print("AI is thinking...")
            # Send prompt to DeepSeek
            # Adjust predict_kwargs as needed for desired output length and creativity
            response_text = model.respond(inference_prompt, predict_kwargs={"temperature": 0.7, "max_tokens": 768})

            # Print the full response, which should include the CoT
            print(f"\nAI Response:\n{response_text}")

        except LMSException as e:
            print(f"Error during inference: {e}")
        except KeyboardInterrupt:
            print("\nExiting inference due to user interrupt.")
            break
        except Exception as e:
            print(f"An unexpected error occurred during inference: {e}")
            break # Exit loop on unexpected error

def main():
    print("--- Starting DeepSeek Inference Script ---")
    client = get_lmstudio_client()
    if not client:
        return

    with client: # Ensure client resources are managed
        model = get_model_reference(client, MODEL_IDENTIFIER)
        if not model:
            print("Could not obtain model reference for inference. Exiting.")
            return

        run_inference_loop(model)

    print("--- DeepSeek Inference Script Finished ---")

if __name__ == "__main__":
    main()
