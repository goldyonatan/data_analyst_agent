from __future__ import annotations
import re
from typing import Optional, Dict, List, Any

# --- simple extractors -------------------------------------------------

def _extract_user_facts(text: str) -> Dict[str, Any]:
    """Pull lightweight facts from user text: name + preferences."""
    facts: Dict[str, Any] = {}
    if not text:
        return facts

    # name: "my name is ...", "call me ..."
    m = re.search(r"(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][\w\-\s]{1,40})", text, re.I)
    if m:
        facts["name"] = m.group(1).strip().rstrip(".,!")

    # examples preference: "show/give me N examples"
    m = re.search(r"(?:show|give)\s+me\s+(\d{1,3})\s+examples", text, re.I)
    if m:
        facts["n_examples_preference"] = max(1, min(100, int(m.group(1))))

    prefs: List[str] = []
    if re.search(r"\b(table|markdown\s+table)\b", text, re.I):
        prefs.append("prefer_table")
    if re.search(r"\bbrief|concise|short\s+answer\b", text, re.I):
        prefs.append("prefer_brief")
    if re.search(r"\bbar\s*chart|chart\b", text, re.I):
        prefs.append("prefer_chart")

    if prefs:
        facts["preferences"] = prefs
    return facts


def _append_recent(seq: List[str], item: Optional[str], maxlen: int = 10) -> List[str]:
    """Append de-duped (move-to-end) with a max length."""
    seq = list(seq or [])
    if not item:
        return seq
    item = str(item)
    if item in seq:
        seq.remove(item)
    seq.append(item)
    if len(seq) > maxlen:
        seq = seq[-maxlen:]
    return seq


# --- memory update & format --------------------------------------------

def update_summary_memory(
    memory: Dict[str, Any],
    *,
    turn_text: str,
    last_intent: Optional[str] = None,
    last_category: Optional[str] = None,
    last_categories_list: Optional[List[str]] = None,
    last_intents_list: Optional[List[str]] = None,
    last_list_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decides what to store after each turn:
    - Name & lightweight preferences (if mentioned)
    - Last selected intent/category + small recent history (de-duped)
    - Last shown lists (for ordinal follow-ups) + kind
    """
    mem = dict(memory or {})

    # facts from this turn
    facts = _extract_user_facts(turn_text or "")
    if facts.get("name"):
        mem["name"] = facts["name"]
    if facts.get("n_examples_preference"):
        mem["n_examples_preference"] = facts["n_examples_preference"]
    if facts.get("preferences"):
        # de-dupe merge
        cur = set(mem.get("preferences", []))
        for p in facts["preferences"]:
            cur.add(p)
        mem["preferences"] = sorted(cur)

    # selections
    if last_intent:
        mem["last_selected_intent"] = last_intent
        mem["recent_intents"] = _append_recent(mem.get("recent_intents", []), last_intent)
    if last_category:
        mem["last_selected_category"] = last_category
        mem["recent_categories"] = _append_recent(mem.get("recent_categories", []), last_category)

    # lists for ordinal reference + kind
    if isinstance(last_categories_list, list):
        mem["last_categories_list"] = list(last_categories_list)
    if isinstance(last_intents_list, list):
        mem["last_intents_list"] = list(last_intents_list)
    if last_list_kind in ("category", "intent"):
        mem["last_list_kind"] = last_list_kind

    return mem


def format_memory(memory: Dict[str, Any]) -> str:
    """Pretty, concise rendering of what we remember."""
    mem = memory or {}
    lines: List[str] = []

    if mem.get("name"):
        lines.append(f"Name: {mem['name']}")
    if mem.get("preferences"):
        pretty = ", ".join(mem["preferences"])
        lines.append(f"Preferences: {pretty}")

    if mem.get("last_selected_intent"):
        lines.append(f"Last intent: {mem['last_selected_intent']}")
    if mem.get("last_selected_category"):
        lines.append(f"Last category: {mem['last_selected_category']}")

    if mem.get("recent_intents"):
        ri = ", ".join(mem["recent_intents"][-2:])
        lines.append(f"Recent intents: {ri}")
    if mem.get("recent_categories"):
        rc = ", ".join(mem["recent_categories"][-2:])
        lines.append(f"Recent categories: {rc}")

    if not lines:
        return "Here's what I remember about you: (nothing yet)."
    return "Here's what I remember about you: " + "; ".join(lines)
