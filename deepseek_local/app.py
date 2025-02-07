import os
import json
import time
import re
import torch
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from deepseek_model import download_model, load_model

# Load settings
SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")


def load_settings():
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def build_prompt_simple(conversation, new_message):
    """
    Build a prompt that:
      - Includes previous conversation history (if any)
      - Instructs the model to consider the complexity of the query and
        adjust the answer length (up to 12,000 tokens) as needed.
      - Instructs the model to output only the final answer without any internal reasoning.
      - Requires the final output to be preceded by 'Final Answer:'.
    """
    # Gather all previous messages (excluding placeholders like "Thinking...")
    history_parts = [
        msg["content"]
        for msg in conversation
        if msg["role"] in ["user", "assistant"] and msg["content"] != "Thinking..."
    ]
    history = " ".join(history_parts)

    if history:
        prompt = (
            f"This is the previous conversation history: {history}. "
            "Analyze it to determine the complexity of the query and decide on an appropriate response length, up to 12,000 tokens if needed. "
            "Now, provide your final answer to the following prompt without showing any internal reasoning or chain-of-thought. "
            "Your output should start with 'Final Answer:' and include only the answer afterward. "
            f"Respond only to: {new_message}"
        )
    else:
        prompt = (
            "Analyze the complexity of the following prompt and provide your final answer accordingly, up to 12,000 tokens if needed. "
            "Do not include any internal reasoning or chain-of-thought; only the final answer should be provided. "
            "Your output must begin with 'Final Answer:'. "
            f"Respond only to: {new_message}"
        )
    return prompt


def extract_final_answer(generated_text):
    """
    Extracts and returns only the final answer by searching for the marker 'Final Answer:'.
    If the marker isn't found, returns the entire generated text.
    """
    match = re.search(r"Final Answer:\s*(.*)", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-very-secret-key"

# Download and load the model once at startup
download_model()
tokenizer, model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.route("/", methods=["GET", "POST"])
def chat():
    settings = load_settings()
    if "conversation" not in session:
        session["conversation"] = []
    conversation = session["conversation"]

    if request.method == "POST":
        user_message = request.form.get("message")
        if user_message:
            # Append the user message and add a placeholder for the assistant reply.
            conversation.append({"role": "user", "content": user_message})
            conversation.append({"role": "assistant", "content": "Thinking..."})
            session["conversation"] = conversation  # update session immediately

            # Build the prompt including history and the new message.
            prompt = build_prompt_simple(conversation, user_message)

            # Tokenize and generate the model's output.
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=settings.get("max_output_tokens", 12000),
                    temperature=0.7,  # Adjusts randomness
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.1  # Reduces repetition
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_response = extract_final_answer(generated)

            # Replace the "Thinking..." placeholder with the final response.
            conversation[-1]["content"] = final_response
            session["conversation"] = conversation

            time.sleep(0.2)  # Small delay for better UX
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"assistant": final_response})
            else:
                return redirect(url_for("chat"))

    return render_template("chat.html", conversation=conversation)


@app.route("/reset")
def reset():
    session.pop("conversation", None)
    return redirect(url_for("chat"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        new_system_prompt = request.form.get("system_prompt", "")
        show_reasoning = request.form.get("show_reasoning", "off") == "on"
        settings = load_settings()
        settings["system_prompt"] = new_system_prompt
        settings["show_reasoning"] = show_reasoning
        save_settings(settings)
        return redirect(url_for("chat"))
    else:
        settings = load_settings()
        return render_template("settings.html", settings=settings)


if __name__ == "__main__":
    app.run(debug=True)
