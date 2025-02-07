import os
import sys
import subprocess
import shutil
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Set the directory where the model will be stored.
MODEL_DIR = os.path.join("models", "deepseek")
# Instead of using the full DeepSeek-R1 repo (671B parameters), we use the distilled smallest model.
HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def is_config_valid():
    """
    Check if the config.json exists in MODEL_DIR and has a valid "model_type" key.
    """
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "model_type" in config:
                return True
            else:
                print("Invalid config.json: missing 'model_type' key.")
        except Exception as e:
            print("Error reading config.json:", e)
    return False

def download_model():
    """
    Ensure the model is available locally.

    If the model directory or its config file is missing or invalid, remove it and
    download the model via Hugging Face auto-download (using the distilled smallest model).
    """
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not (os.path.exists(MODEL_DIR) and os.path.exists(config_path) and is_config_valid()):
        if os.path.exists(MODEL_DIR):
            print("Removing invalid or incomplete model directory...")
            shutil.rmtree(MODEL_DIR)
        print("Downloading model via Hugging Face...")
        # Load configuration (and remove any quantization settings if present)
        config = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            config.quantization_config = None
        # Download tokenizer and model using the HF auto-download API.
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        os.makedirs(MODEL_DIR, exist_ok=True)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print("Model downloaded via Hugging Face and saved to", MODEL_DIR)
    else:
        print("Model already downloaded and valid at", MODEL_DIR)

def load_model():
    """
    Load the model and tokenizer from the local directory.

    Returns:
      tokenizer: The AutoTokenizer instance.
      model: The AutoModelForCausalLM instance loaded on the appropriate device.
    """
    print("Loading model and tokenizer from", MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        config.quantization_config = None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    # Use CUDA if available, else fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model

def generate_text(system_prompt, assistant_prompt, user_prompt, max_tokens=100):
    """
    Combine the given prompts and generate text using the DeepSeek model.

    Parameters:
      system_prompt (str): The base system prompt.
      assistant_prompt (str): The assistant's context or previous message.
      user_prompt (str): The user's input prompt.
      max_tokens (int): Maximum number of tokens to generate.

    Returns:
      response (str): The generated text.
      tokens_used (int): The number of tokens in the generated response.
    """
    tokenizer, model = load_model()  # In production, consider loading these once globally.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_prompt = f"{system_prompt}\n{assistant_prompt}\n{user_prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=max_tokens)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_used = len(tokenizer.encode(response))
    return response, tokens_used

if __name__ == "__main__":
    # Determine the target directory (pass as a command-line argument or use current directory)
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = os.getcwd()  # Use current directory if none provided

    # Define project name and directories
    project_name = "deepseek_local"
    project_dir = os.path.join(target_directory, project_name)
    models_dir = os.path.join(project_dir, "models")
    deepseek_dir = os.path.join(models_dir, "deepseek")

    # Create the directory structure
    os.makedirs(deepseek_dir, exist_ok=True)
    print(f"Created directories:\n  {project_dir}\n  {models_dir}\n  {deepseek_dir}")

    # 1. Create requirements.txt
    requirements_content = """torch
transformers
"""
    requirements_path = os.path.join(project_dir, "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("Created requirements.txt")

    # 2. Create deepseek_model.py (this file content is the current script)
    deepseek_model_path = os.path.join(project_dir, "deepseek_model.py")
    with open(deepseek_model_path, "w", encoding="utf-8") as f:
        f.write(
            "# This file defines functions to download, load, and use the DeepSeek model locally.\n"
            + open(__file__, "r", encoding="utf-8").read()
        )
    print("Created deepseek_model.py")

    # 3. Create main.py that uses deepseek_model.py
    main_content = r'''from deepseek_model import download_model, generate_text

def main():
    # Ensure the model is downloaded locally.
    download_model()

    # Define the prompts.
    system_prompt = "System: You are a helpful AI assistant."
    assistant_prompt = "Assistant: How can I assist you today?"
    user_prompt = "Hey, can you tell me about DeepSeek's smallest model?"

    # Generate the response.
    response, tokens = generate_text(system_prompt, assistant_prompt, user_prompt, max_tokens=150)

    # Print the output.
    print("Generated Response:")
    print(response)
    print("\nTokens used:", tokens)

if __name__ == "__main__":
    main()
'''
    main_path = os.path.join(project_dir, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    print("Created main.py")

    print("\nProject setup complete in:", project_dir)
    print("To continue, navigate to the project folder and install dependencies with:")
    print("    pip install -r requirements.txt")
    print("Then run the sample script with:")
    print("    python main.py")
