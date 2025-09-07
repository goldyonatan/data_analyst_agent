from __future__ import annotations
from typing import Literal, TypedDict, Annotated
import re
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from .models import get_chat_model


class RoutePrediction(BaseModel):
    reasoning: str
    route: Literal["structured", "unstructured", "oos", "memory_query"]


class RouterState(TypedDict):
    user_message: str
    route: str
    messages: Annotated[list, add_messages]


ROUTER_SYSTEM = """You are a router for a Bitext data analyst agent.
Decide where to send the user request:
- "structured" for counting, listing, examples, distributions (tools will be used)
- "unstructured" for free-form summaries/analysis over dataset slices
- "memory_query" only if the user explicitly asks what you remember about them/the conversation
- "oos" for anything not about the dataset

Return a route + a short reasoning.
"""


# ---- Keyword sets ----
DATASET_HINTS = {
    "category", "categories", "intent", "intents", "example", "examples",
    "distribution", "count", "how many", "list", "top", "most frequent",
    # dataset domain words:
    "refund", "order", "account", "invoice", "payment", "delivery",
    "shipping", "subscription", "contact", "feedback",
    # dataset names:
    "bitext", "dataset",
}

STRUCTURED_TRIGGERS = {
    "example", "examples", "category", "categories", "intent", "intents",
    "distribution", "histogram", "most frequent", "top", "count",
    "how many", "list", "show me more", "show examples",
}

# strict memory detection: must contain 'remember' and a self-reference
MEMORY_RE = re.compile(
    r"\b(what\s+do\s+you\s+remember(?:\s+about)?\s+(?:me|us|our|this)|"
    r"what\s+can\s+you\s+remember(?:\s+about)?\s+(?:me|us|our|this)|"
    r"remember\s+(?:me|us|our|this)\b)", re.IGNORECASE
)


def _contains_any(text: str, needles: set[str]) -> bool:
    lt = text.lower()
    return any(n in lt for n in needles)


def classify_query(state: RouterState):
    text = (state["user_message"] or "")
    lt = text.lower()

    # 1) STRICT memory: only if they explicitly ask what you remember
    if MEMORY_RE.search(lt):
        route = "memory_query"
        reasoning = "explicit memory question"
        return {
            "route": route,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": state["user_message"]},
                {"role": "assistant", "content": f"[route={route}] {reasoning}"},
            ],
        }

    # 2) Structured override: obvious dataset operations
    if _contains_any(lt, STRUCTURED_TRIGGERS):
        route = "structured"
        reasoning = "keyword override → structured"
        return {
            "route": route,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": state["user_message"]},
                {"role": "assistant", "content": f"[route={route}] {reasoning}"},
            ],
        }

    # 3) If there are no dataset hints at all → OOS (e.g., “What is Serj’s rating?”)
    if not _contains_any(lt, DATASET_HINTS):
        route = "oos"
        reasoning = "no dataset-related hints → out of scope"
        return {
            "route": route,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": state["user_message"]},
                {"role": "assistant", "content": f"[route={route}] {reasoning}"},
            ],
        }

    # 4) Otherwise let the model decide
    chat = get_chat_model(temperature=0.0)
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": state["user_message"]},
    ]
    try:
        resp = chat.with_structured_output(RoutePrediction).invoke(messages)
        route = resp.route
        reasoning = resp.reasoning
    except Exception as e:
        route = "oos"
        reasoning = f"fallback (error: {e.__class__.__name__})"

    return {
        "route": route,
        "messages": messages + [{"role": "assistant", "content": f"[route={route}] {reasoning}"}],
    }
