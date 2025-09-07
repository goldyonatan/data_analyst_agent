from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius

load_dotenv()
DEFAULT_MODEL = os.getenv("NEBIUS_MODEL", "Qwen/Qwen2.5-32B-Instruct")

def get_chat_model(temperature: float = 0.2) -> ChatNebius:
    return ChatNebius(
        model=DEFAULT_MODEL,
        temperature=temperature,
        top_p=0.95,
    )
