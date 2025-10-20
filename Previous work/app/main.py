# app/main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.prompt import build_prompt, CHARACTERS
from app.schemas import DialogueRequest, DialogueResponse


load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")  # fallback

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not set in environment")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

app = FastAPI(title="Questify Dialogue API")

def call_hf_inference(prompt: str, max_tokens: int = 80) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
            # add other model-specific params if needed
        },
        "options": {"wait_for_model": True}
    }
    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=60)
    if resp.status_code != 200:
        # try to return body for debugging
        raise HTTPException(status_code=502, detail=f"Hugging Face API error: {resp.status_code} {resp.text}")
    data = resp.json()
    # HF can return multiple formats; try to capture common ones:
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=502, detail=f"HF error: {data['error']}")
    # If list of dicts with generated_text
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # Some models return plain text directly (string)
    if isinstance(data, str):
        return data.strip()
    # other nested shapes: try to find any string
    # fallback to stringify
    return str(data).strip()

@app.post("/generate", response_model=DialogueResponse)
def generate_dialogue(req: DialogueRequest):
    if req.character.lower() not in CHARACTERS:
        raise HTTPException(status_code=400, detail="Invalid character")
    prompt = build_prompt(req.character.lower(), req.task, req.deadline, req.context)
    try:
        raw = call_hf_inference(prompt)
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference call failed: {e}")

    # Postprocess: keep only first line, trim quotes/spaces
    cleaned = raw.strip().splitlines()[0].strip()
    # Optionally remove leading "CharacterName:" if model includes it
    for name in [c['name'] for c in CHARACTERS.values()]:
        if cleaned.lower().startswith(name.lower() + ":"):
            cleaned = cleaned.split(":", 1)[1].strip()
    return {"dialogue": cleaned}
