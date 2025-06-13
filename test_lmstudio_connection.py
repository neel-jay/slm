import os
from lmstudio import Client as LMSClient # Corrected import
from lmstudio.sdk_api import LMStudioError as LMSException # Corrected import for exceptions

# Try to get the model identifier from an environment variable,
# otherwise, use the one provided by the user.
# This makes the script slightly more flexible for testing in different environments.
MODEL_IDENTIFIER = os.getenv("LMSTUDIO_MODEL_IDENTIFIER", "deepseek/deepseek-r1-0528-qwen3-8b")

def test_connection():
    print(f"Attempting to connect to LM Studio and model: {MODEL_IDENTIFIER}")
    try:
        with LMSClient() as client:
            print("Successfully created LMSClient.")

            # List loaded models - useful for debugging
            try:
                loaded_models = client.list_loaded_models()
                if loaded_models:
                    print("\n--- Loaded Models in LM Studio ---")
                    for i, model_info in enumerate(loaded_models):
                        print(f"{i+1}. Name: {model_info.name}, Path: {model_info.path}, ID: {model_info.id}")
                        # Check if any loaded model's path or name contains the target identifier
                        if MODEL_IDENTIFIER.lower() in model_info.path.lower() or \
                           MODEL_IDENTIFIER.lower() in model_info.name.lower():
                            print(f"   ^-- This looks like a potential match for '{MODEL_IDENTIFIER}'")
                    print("------------------------------------\n")
                else:
                    print("No models currently loaded in LM Studio according to the SDK.")
            except LMSException as e:
                print(f"Could not list loaded models: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while listing models: {e}")


            print(f"Attempting to get model handle for: {MODEL_IDENTIFIER}")
            # Corrected model access: client.llm is a session, then call .model() on it
            llm_handle = client.llm.model(MODEL_IDENTIFIER)
            print(f"Successfully got model handle: {llm_handle.identifier}")

            prompt = "Hello! Who are you? Respond in one short sentence."
            print(f"Sending prompt: '{prompt}'")

            try:
                # Corrected method call: llm_handle.respond()
                # predict_kwargs becomes 'config' in the new API
                prediction_result = llm_handle.respond(prompt, config={"temperature": 0.7, "max_tokens": 50})

                # Corrected response parsing based on sync_api.py structure for PredictionResult
                if prediction_result and prediction_result.choices and len(prediction_result.choices) > 0:
                    if hasattr(prediction_result.choices[0], 'message') and \
                       hasattr(prediction_result.choices[0].message, 'content'):
                        response_content = prediction_result.choices[0].message.content
                        print(f"Response content: {response_content}")
                    else:
                        print(f"Response structure unexpected (message or content missing): {prediction_result.choices[0]}")
                else:
                    print(f"Received unexpected response structure (no choices or empty choices): {prediction_result}")

            except LMSException as e:
                print(f"Error during llm_handle.respond: {e}") # Keep LMSException for now
            except Exception as e:
                print(f"An unexpected error occurred during llm_handle.respond: {e}")

    except LMSException as e: # Keep LMSException for now
        print(f"LM Studio SDK Exception: {e}")
        print("Please ensure LM Studio is running, the server is enabled (usually on http://localhost:1234),")
        print(f"and the model '{MODEL_IDENTIFIER}' is available and correctly identified.")
    except ConnectionRefusedError:
        print("Connection Refused: Could not connect to LM Studio server. Is it running and server enabled?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_connection()
