# app/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

Character = Literal["liya", "mom", "mrs_smith", "harry", "mentor"]
Context = Literal["initial", "followup"]

class DialogueRequest(BaseModel):
    character: Character = Field(..., description="Which character voice to use")
    task: str = Field(..., description="Task title or short description")
    deadline: str = Field(..., description="Deadline as ISO date or human phrase")
    context: Context = Field(..., description="'initial' or 'followup'")

class DialogueResponse(BaseModel):
    dialogue: str
