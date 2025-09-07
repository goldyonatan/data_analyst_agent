from __future__ import annotations
from langchain_core.tools import tool

@tool
def summarize(user_request: str) -> str:
    """Summarize behavior or patterns in natural language based on dataset examples.
    Input: `user_request` — a free-form question asking for a summary/analysis.
    Output: A marker string "summarize_request: <request>" that the unstructured agent will expand.
    """
    return f"summarize_request: {user_request}"
