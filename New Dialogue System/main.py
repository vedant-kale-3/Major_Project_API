# main.py
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype="auto"
)

def resolve_tone(urgency, character_tone):
    urgency_tone_map = {
        "Low": "gentle and relaxed",
        "Medium": "reminding and slightly serious",
        "High": "urgent but caring",
        "Critical": "very urgent and strict"
    }
    base_tone = urgency_tone_map.get(urgency, "motivating")
    if character_tone:
        return f"{base_tone}, with a {character_tone} personality"
    return base_tone

@app.route("/")
def home():
    return jsonify({"status": "Questify AI API Running"})

@app.route("/generate-message", methods=["POST"])
def generate_message():
    data = request.json
    required_fields = ["task", "character", "urgency"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    task = data["task"]
    character = data["character"]
    urgency = data["urgency"]
    character_tone = data.get("character_tone", "")
    tone = resolve_tone(urgency, character_tone)

    prompt = (
        f"Generate a short motivational reminder.\n"
        f"Character: {character}\nTask: {task}\nUrgency: {urgency}\n"
        f"Tone: {tone}\nKeep it under 2 lines. Speak directly to the user.\nMessage:"
    )

    result = generator(prompt, max_new_tokens=30, do_sample=True,
                       temperature=0.85, top_p=0.9,
                       pad_token_id=generator.tokenizer.eos_token_id)
    generated = result[0]["generated_text"][len(prompt):].strip()

    return jsonify({"character": character, "task": task,
                    "urgency": urgency, "tone": tone, "message": generated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)