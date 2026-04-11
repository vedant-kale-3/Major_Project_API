from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

API_URL = "https://router.huggingface.co/v1/chat/completions"

def generate_text(prompt):
    # Read token inside the function so it's always fresh
    hf_token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN loaded: {hf_token[:10] if hf_token else 'NOT FOUND'}")

    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct:novita",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 60,
            "temperature": 0.85,
            "top_p": 0.9
        }
    )

    print(f"Status Code: {response.status_code}")
    print(f"Raw Response: {response.text}")

    if not response.text.strip():
        return f"Error: Empty response from HuggingFace (status {response.status_code})"

    result = response.json()

    if "error" in result:
        return f"Model error: {result['error']}"

    return result["choices"][0]["message"]["content"].strip()

def resolve_tone(urgency, character_tone):
    urgency_tone_map = {
        "Low":      "gentle and relaxed",
        "Medium":   "reminding and slightly serious",
        "High":     "urgent but caring",
        "Critical": "very urgent and strict"
    }
    base_tone = urgency_tone_map.get(urgency, "motivating")
    if character_tone:
        return f"{base_tone}, with a {character_tone} personality"
    return base_tone

@app.route("/")
def home():
    return jsonify({"status": "Questify AI API is running"})

@app.route("/generate-message", methods=["POST"])
def generate_message():
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required_fields = ["task", "character", "urgency"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    task           = data["task"]
    character      = data["character"]
    urgency        = data["urgency"]
    character_tone = data.get("character_tone", "")
    tone           = resolve_tone(urgency, character_tone)

    prompt = (
        f"Generate a short motivational reminder.\n"
        f"Character: {character}\n"
        f"Task: {task}\n"
        f"Urgency: {urgency}\n"
        f"Tone: {tone}\n"
        f"Keep it under 2 lines. Speak directly to the user.\n"
        f"Message:"
    )

    message = generate_text(prompt)

    return jsonify({
        "character": character,
        "task":      task,
        "urgency":   urgency,
        "tone":      tone,
        "message":   message
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)