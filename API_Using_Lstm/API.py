from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Motivational Dialogue Generator API", version="2.0.0")

OPENROUTER_API_KEY = "paste_your_openrouter_key_here"

SYSTEM_PROMPT = """You are a motivational coach in a gamified productivity app called Questify.
Follow these rules STRICTLY:
1. Speak like a cool, energetic film director / creative mentor
2. Always wrap the task name in **bold markdown**
3. Use cinematic metaphors (filming, directing, editing, etc.)
4. Maximum 2 sentences only. Never write more.
5. Always reference the specific task the user gives you
6. Output ONLY the motivational line. No intro, no explanation, no emojis."""

SITUATION_CONTEXT = {
    "first_time":   "The user is attempting this task for the FIRST TIME. Be extra encouraging.",
    "notification": "The user MISSED their deadline. Be urgent but supportive, not harsh.",
    "general":      "The user needs a motivational push to start their task.",
}

TASK_COMPLETED_RESPONSE = "Mission complete — your strength echoes in the halls of Questify!"

# Ordered fallback list — all confirmed active free models March 2026
MODELS = [
    "openrouter/auto",                              # auto-picks best available free model
    "meta-llama/llama-3.1-405b-instruct:free",      # Meta 405B — massive, GPT-4 level
    "mistralai/mistral-small-3.1-24b-instruct:free",# Mistral — great instruction following
    "qwen/qwen3-30b-a3b:free",                      # Qwen3 — strong multilingual
]

class TaskRequest(BaseModel):
    Task: str
    Situation: str

def call_llm(task: str, situation_context: str):
    for model in MODELS:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Task: {task}\nContext: {situation_context}"}
                ],
                "max_tokens": 80,
                "temperature": 0.85
            }
        )

        data = response.json()
        print(f"[{model}] Response:", data)

        if "choices" in data:
            dialogue = data["choices"][0]["message"]["content"].strip()
            sentences = dialogue.split(". ")
            if len(sentences) > 2:
                dialogue = ". ".join(sentences[:2]) + "."
            return dialogue

        print(f"[{model}] Failed, trying next model...")

    return "Your scene is set — now step into the frame and make it happen."

@app.post("/generate-dialogue/")
def generate_dialogue(req: TaskRequest):
    situation = req.Situation.lower().strip()

    if situation == "done":
        return {"Dialogue": TASK_COMPLETED_RESPONSE}

    situation_context = SITUATION_CONTEXT.get(
        situation,
        "The user needs general motivation for their task."
    )

    dialogue = call_llm(req.Task, situation_context)
    return {"Dialogue": dialogue}
