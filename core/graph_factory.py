from __future__ import annotations
from typing import TypedDict, Annotated, Optional, Literal, List
from rapidfuzz import process as rf_process, fuzz as rf_fuzz  # add near other imports
import os, re

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .models import get_chat_model
from .dataset import load_bitext_dataframe, find_examples
from .tools_structured import build_structured_tools
from .tools_unstructured import summarize as summarize_tool
from .routers import classify_query
from .memory import update_summary_memory

DEBUG = os.getenv("DEBUG_AGENT", "0") == "1"


class AgentState(TypedDict, total=False):
    thread_id: str
    user_message: str
    route: str
    messages: Annotated[list, add_messages]
    final_response: str
    selected_intent: Optional[str]
    selected_category: Optional[str]
    last_categories_list: List[str]
    last_intents_list: List[str]
    last_list_kind: Optional[str]
    summary_memory: dict


def build_graph(checkpointer):
    df = load_bitext_dataframe()

    (
        structured_tools,
        get_selection,
        set_selection,
        get_listing_state,
        set_listing_state,
        ALL_INTENTS,
        ALL_CATEGORIES,
    ) = build_structured_tools(df)
    tool_node_structured = ToolNode(structured_tools, name="structured_tools")

    unstructured_tools = [summarize_tool]           # keep node around, but we won't bind tools
    tool_node_unstructured = ToolNode(unstructured_tools, name="unstructured_tools")
    unstructured_llm = get_chat_model(temperature=0.2)  # <-- NO bind_tools here
    structured_llm = get_chat_model(temperature=0.0).bind_tools(structured_tools, parallel_tool_calls=False)
    finalizer_llm = get_chat_model(temperature=0.0)  # no tools

    graph = StateGraph(AgentState)

    # ---------- Helpers ----------
    def _persisted_selection(state: AgentState):
        sm = state.get("summary_memory") or {}
        intent = state.get("selected_intent") or sm.get("last_selected_intent")
        category = state.get("selected_category") or sm.get("last_selected_category")
        if DEBUG:
            print(f"[seed] selection intent={intent} category={category}")
        return intent, category

    def _persisted_listings(state: AgentState):
        sm = state.get("summary_memory") or {}
        cats = state.get("last_categories_list") or sm.get("last_categories_list") or []
        ints = state.get("last_intents_list") or sm.get("last_intents_list") or []
        last_kind = state.get("last_list_kind") or sm.get("last_list_kind")
        if DEBUG:
            print(f"[seed] listings cats={len(cats)} intents={len(ints)} last_kind={last_kind}")
        return cats, ints, last_kind

    def _latest_tool_text(messages: list) -> str:
        for m in reversed(messages):
            content = getattr(m, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        return ""
    
    def _maybe_match_with_score(text: str, choices: list[str], min_score: int = 65):
        """
        Case-insensitive partial match against choices.
        Returns (canonical_choice, score) or (None, 0).
        """
        if not text or not choices:
            return (None, 0)
        text_l = text.lower()
        # build a lower->canonical map so we can return the original casing
        lowered = [c.lower() for c in choices]
        res = rf_process.extractOne(text_l, lowered, scorer=rf_fuzz.partial_ratio)
        if not res:
            return (None, 0)
        choice_l, score, idx = res  # RapidFuzz returns (choice, score, index)
        if score < min_score:
            return (None, score or 0)
        # map back to canonical (original) value by index
        return (choices[idx], score)

    # ---------- Nodes ----------
    def router_node(state: AgentState):
        return classify_query({"user_message": state["user_message"], "route": "", "messages": []})

    def structured_agent(state: AgentState):
        intent, category = _persisted_selection(state)
        cats, ints, last_kind = _persisted_listings(state)
        set_selection(intent, category)
        set_listing_state(cats, ints, last_kind)
        if DEBUG:
            print(f"[structured_agent] user={state.get('user_message')!r} seed intent={intent} category={category}")

        system = f"""
You are a data analyst agent for the Bitext Customer Service dataset.
Fields: flags, instruction, category, intent, response.

Tool policy:
- If the user does NOT name a new intent/category, KEEP USING the current selection
  (intent={intent!r}, category={category!r}) and DO NOT call select_* again.
- For "show examples", call show_examples(n) with the current selection.
- For "what categories exist", call list_categories().
- For "what intents exist", call list_intents().
- For "most frequent categories" / "category distribution", call top_categories(n).
- For "most frequent intents" / "intent distribution", call top_intents(n).
- For "show examples of the most frequent categories", call examples_for_top_categories(k=5, n_per=3).
- For "show examples of the most frequent intents", call examples_for_top_intents(k=5, n_per=3).
- For "intent distribution", call top_intents(n=10, as_markdown=True, with_bars=True).
- For "category distribution", call top_categories(n=10, as_markdown=True, with_bars=True).
- For "total count of the last N intents/categories", call count_last_from_previous(kind="auto", n=N, include_breakdown=True, as_markdown=True).
- If the user says "the Nth", "#N", or "number N", call select_from_previous(kind="auto", which="nth", n=N) then call show_examples().
- If the user refers to "the last one on your list" or similar, call select_from_previous(kind="auto", which="last") then call show_examples().
- When finished, call finish().

Available categories (sample): {ALL_CATEGORIES[:15]} ... (total={len(ALL_CATEGORIES)})
Available intents   (sample): {ALL_INTENTS[:15]} ... (total={len(ALL_INTENTS)})
"""
        user = state["user_message"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp = structured_llm.invoke(msgs)
        return {"messages": msgs + [resp]}

    def unstructured_agent(state: AgentState):
        # 1) start with persisted selection
        intent, category = _persisted_selection(state)

        # 2) if the user text names a known intent/category, OVERRIDE the seed selection
        user_text = (state.get("user_message") or "")
        # These lists come from build_structured_tools(...) earlier in build_graph
        # (keep ALL_INTENTS, ALL_CATEGORIES in the outer scope of build_graph)
        nonlocal ALL_INTENTS, ALL_CATEGORIES  # if defined in the outer build_graph scope

        matched_intent, si = _maybe_match_with_score(user_text, ALL_INTENTS, min_score=70)
        matched_category, sc = _maybe_match_with_score(user_text, ALL_CATEGORIES, min_score=70)

        # Preference: if the user names a category, use it. If they name an intent, use that.
        # If both are present, keep both (narrow slice). If neither, fall back to persisted.
        if matched_category:
            category = matched_category
        if matched_intent:
            intent = matched_intent

        # 3) Pull context examples for the chosen slice
        examples_df = find_examples(
            load_bitext_dataframe(),
            intent=intent,
            category=category,
            n=12,
        )
        context_lines = [
            f"- [{row['intent']} / {row['category']}] {row['instruction']} -> {row['response']}"
            for _, row in examples_df.iterrows()
        ]
        context = "\n".join(context_lines) or "(no examples)"

        # Make target label human-friendly
        target = intent if intent else category if category else "the selected slice"

        system = (
            "You are a senior data analyst. Read the examples and write a concise, factual summary.\n"
            "Output format:\n"
            "1) One sentence overview.\n"
            "2) 3–6 bullets: patterns, common user asks, agent behaviors, pitfalls.\n"
            "3) If examples are empty, say so briefly.\n"
            "Do not invent data; use only the context."
        )
        user = f"""Summarize {target} based on the examples below.

    Context examples:
    {context}
    """

        if DEBUG:
            print(f"[unstructured_agent] summarizing target={target} "
                f"(examples={examples_df.shape[0] if hasattr(examples_df,'shape') else 0}) "
                f"overrides: intent={matched_intent}({si}), category={matched_category}({sc})")

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp = unstructured_llm.invoke(msgs)

        # Persist what we actually summarized so memory has the right last_* values this turn
        return {
            "messages": msgs + [resp],
            "selected_intent": intent,
            "selected_category": category,
        }

    def out_of_scope(state: AgentState):
        return {"final_response": "Sorry — that request is out of scope for this dataset. Try intents, categories, counts, or summaries."}

    def pre_summarize_selector(state: AgentState):
        """
        Resolve 'the 7th', '#6', 'last', etc. against the most recently listed items,
        set selected_category/intent accordingly, then return (no text yet).
        """
        text = (state.get("user_message") or "").lower()

        # Pull last lists/kind from state or summary memory (hydrated by app.py)
        sm = state.get("summary_memory") or {}
        cats = state.get("last_categories_list") or sm.get("last_categories_list") or []
        ints = state.get("last_intents_list") or sm.get("last_intents_list") or []
        last_kind = state.get("last_list_kind") or sm.get("last_list_kind")

        # Decide kind
        if "intent" in text:
            kind = "intent"
        elif "category" in text or "categories" in text:
            kind = "category"
        else:
            kind = last_kind or ("category" if cats else "intent" if ints else None)

        # Index to pick
        idx = None
        if "last" in text or "final" in text:
            # last item
            seq_len = len(cats) if kind == "category" else len(ints)
            idx = seq_len - 1 if seq_len else None
        elif "first" in text:
            idx = 0
        else:
            m = re.search(r"(?:#|no\.?|number)?\s*(\d+)", text)
            if m:
                n = max(1, int(m.group(1)))
                idx = n - 1

        # Apply selection
        selected_intent = state.get("selected_intent")
        selected_category = state.get("selected_category")
        if kind == "category" and cats and idx is not None and 0 <= idx < len(cats):
            selected_category = cats[idx]
        elif kind == "intent" and ints and idx is not None and 0 <= idx < len(ints):
            selected_intent = ints[idx]

        if DEBUG:
            print(f"[pre_summarize_selector] kind={kind} idx={idx} -> "
                f"category={selected_category} intent={selected_intent} "
                f"(cats={len(cats)} ints={len(ints)} last_kind={last_kind})")

        return {
            "selected_category": selected_category,
            "selected_intent": selected_intent,
            # pass through lists/kind so downstream keeps them alive
            "last_categories_list": cats,
            "last_intents_list": ints,
            "last_list_kind": last_kind,
        }

    def after_tools_structured(state: AgentState):
        # Persist selections + listings for follow-ups
        sel = get_selection()
        listings = get_listing_state()
        latest_tool = _latest_tool_text(state["messages"])

        if DEBUG:
            listing_sizes = {key: (len(val) if hasattr(val, "__len__") else "na")
                            for key, val in (listings or {}).items()}
            print(f"[after_structured] latest_tool[:200]={latest_tool[:200]!r} sel={sel} listings={listing_sizes}")

        # --- NEW: If the tool already produced presentational markdown, return it verbatim ---
        lt = (latest_tool or "").lstrip()
        is_markdown_table = lt.startswith("|") and "\n|---" in lt
        looks_like_examples = lt.startswith("### ") or lt.startswith("- ") or lt.startswith("1. ")
        if is_markdown_table or looks_like_examples:
            text = latest_tool or "(no result)"
            return {
                "selected_intent": sel.get("intent"),
                "selected_category": sel.get("category"),
                "last_categories_list": listings.get("last_categories_list", []),
                "last_intents_list": listings.get("last_intents_list", []),
                "last_list_kind": listings.get("last_list_kind"),
                "final_response": text,
            }

        # Otherwise, synthesize a plain-language summary from tool output
        msgs = list(state["messages"]) + [
            {
                "role": "system",
                "content": (
                    "You have the tool results above. Compose the final answer in plain language. "
                    "If the last tool returned examples or a Markdown table, you MUST return it EXACTLY as-is "
                    "(no extra wording). Otherwise, summarize clearly. "
                    "Do NOT call tools. Do NOT reply with 'done'."
                ),
            },
        ]
        if latest_tool:
            msgs.append({"role": "user", "content": f"Last tool output:\n{latest_tool}"})

        resp = finalizer_llm.invoke(msgs)
        text = getattr(resp, "content", "") or latest_tool or "(no result)"
        return {
            "selected_intent": sel.get("intent"),
            "selected_category": sel.get("category"),
            "last_categories_list": listings.get("last_categories_list", []),
            "last_intents_list": listings.get("last_intents_list", []),
            "last_list_kind": listings.get("last_list_kind"),
            "final_response": text,
        }

    def after_tools_unstructured(state: AgentState):
        last = state["messages"][-1]
        content = getattr(last, "content", "") or ""
        return {"final_response": content if isinstance(content, str) else str(content)}

    def memory_summary_node(state: AgentState):
        from .memory import update_summary_memory  # (or top-level import)

        updated = update_summary_memory(
            state.get("summary_memory", {}),
            turn_text=state.get("user_message", ""),
            last_intent=state.get("selected_intent"),
            last_category=state.get("selected_category"),
            last_categories_list=state.get("last_categories_list"),
            last_intents_list=state.get("last_intents_list"),
            last_list_kind=state.get("last_list_kind"),
        )
        if DEBUG:
            print("[summary] updated keys:", sorted(updated.keys()))
        return {"summary_memory": updated}

    def memory_answer(state: AgentState):
        from .memory import format_memory  # (or top-level import)
        return {"final_response": format_memory(state.get("summary_memory") or {})}

    # ---------- Graph wiring ----------
    graph.add_node("router", router_node)
    graph.add_node("pre_summarize_selector", pre_summarize_selector)
    graph.add_node("structured_agent", structured_agent)
    graph.add_node("structured_tools", tool_node_structured)
    graph.add_node("after_structured", after_tools_structured)
    graph.add_node("unstructured_agent", unstructured_agent)
    graph.add_node("unstructured_tools", tool_node_unstructured)
    graph.add_node("after_unstructured", after_tools_unstructured)
    graph.add_node("memory_answer", memory_answer)
    graph.add_node("oos", out_of_scope)
    graph.add_node("summary", memory_summary_node)

    graph.add_edge(START, "router")

    def route_decider(state: AgentState) -> Literal["structured", "unstructured", "memory_query", "oos"]:
        return state["route"]

    graph.add_conditional_edges(
        "router",
        route_decider,
        {
            "structured": "structured_agent",
            "unstructured": "unstructured_agent",
            "summary_from_previous": "pre_summarize_selector",  # NEW
            "memory_query": "memory_answer",
            "oos": "oos",
        },
    )

    graph.add_edge("pre_summarize_selector", "unstructured_agent")
    graph.add_edge("structured_agent", "structured_tools")
    graph.add_edge("structured_tools", "after_structured")
    graph.add_edge("after_structured", "summary")
    graph.add_edge("summary", END)

    graph.add_edge("unstructured_agent", "unstructured_tools")
    graph.add_edge("unstructured_tools", "after_unstructured")
    graph.add_edge("after_unstructured", "summary")

    graph.add_edge("memory_answer", "summary")
    graph.add_edge("oos", "summary")

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled
