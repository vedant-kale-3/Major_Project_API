# app/prompts.py
from typing import Literal
from datetime import datetime

CHARACTERS = {
    "liya": {
        "name": "Liya",
        "persona": "a modern, approachable senior girl who is encouraging and resourceful. Short friendly style."
    },
    "mom": {
        "name": "Mom",
        "persona": "a caring, practical mom who gives gentle guidance and reminders. Nurturing tone."
    },
    "mrs_smith": {
        "name": "Mrs. Smith",
        "persona": "a firm but kind teacher who is constructive and motivates towards learning and discipline."
    },
    "harry": {
        "name": "Harry",
        "persona": "a hard-working, no-nonsense colleague who focuses on getting things done. Straightforward, motivating."
    },
    "mentor": {
        "name": "Mentor",
        "persona": "a wise mentor giving strategic advice and confidence-building feedback. Calming and authoritative."
    }
}

def build_prompt(character_key: str, task: str, deadline: str, context: str) -> str:
    """
    context: 'initial' (task just added) or 'followup' (deadline near)
    """
    char = CHARACTERS.get(character_key.lower())
    if not char:
        raise ValueError("Unknown character")

    # Normalize deadline to human-readable — try parsing ISO first; fallback to raw string
    try:
        dt = datetime.fromisoformat(deadline)
        deadline_h = dt.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        deadline_h = deadline

    if context == "initial":
        instruction = (
            "A user has just added a task. Write a **single short** motivational sentence (max 28 words) "
            "that the character would say to encourage starting the task. "
            "Make it personal and action-oriented. End with one exclamation or emoji if appropriate."
        )
    else:  # followup
        instruction = (
            "The task's deadline is near. Write a **short urgent** motivational reminder (max 30 words) "
            "that pushes the user to take immediate action without being too harsh."
        )

    prompt = (
        f"You are {char['name']}. Persona: {char['persona']}\n\n"
        f"Task: \"{task}\"\n"
        f"Deadline: {deadline_h}\n"
        f"Context: {context}\n\n"
        f"{instruction}\n\n"
        "Return only the message text (no meta, no prefix like 'Liya:')."
    )
    return prompt
