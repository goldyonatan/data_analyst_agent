from __future__ import annotations
from typing import Optional, List
from langchain_core.tools import tool
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import os
import pandas as pd

from .dataset import (
    get_categories,
    get_intents,
    count_by_category,
    count_by_intent,
    find_examples,
)

DEBUG = os.getenv("DEBUG_AGENT", "0") == "1"

def _distribution(df: pd.DataFrame, col: str, n: int):
    s = df[col].dropna().astype(str).str.strip().value_counts()
    total = int(s.sum())
    top = s.head(max(1, int(n)))
    other = int(s.iloc[int(n):].sum()) if len(s) > int(n) else 0
    rows = []
    cum_pct = 0.0
    for i, (name, count) in enumerate(top.items(), start=1):
        pct = (count / total * 100.0) if total else 0.0
        cum_pct += pct
        rows.append((i, name, int(count), pct, cum_pct))
    if other > 0:
        pct = (other / total * 100.0) if total else 0.0
        cum_pct += pct
        rows.append(("…", "Other", other, pct, cum_pct))
    return rows, total

def _bar(pct: float, width: int = 20) -> str:
    # simple ASCII bar scaled to width
    blocks = int(round((pct / 100.0) * width))
    return "█" * blocks

def _to_markdown_table(rows, total: int, header: str, with_bars: bool = True) -> str:
    lines = [f"| # | {header} | Count | % | Cum % | {'Bar' if with_bars else ''}|",
             f"|---|---|---:|---:|---:|{'---:' if with_bars else ''}|"]
    for i, name, count, pct, cum in rows:
        bar = _bar(pct) if with_bars else ""
        lines.append(f"| {i} | {name} | {count} | {pct:.1f}% | {cum:.1f}% | {bar} |" if with_bars
                     else f"| {i} | {name} | {count} | {pct:.1f}% | {cum:.1f}% |")
    lines.append(f"\n_Total rows: **{total}**_")
    return "\n".join(lines)
    
class SelectionState:
    def __init__(self):
        self.selected_intent: Optional[str] = None
        self.selected_category: Optional[str] = None
        self.last_categories_list: List[str] = []
        self.last_intents_list: List[str] = []
        self.last_list_kind: Optional[str] = None  # "category" | "intent"


def _closest(name: str | None, options: List[str]) -> str | None:
    if not name or not options:
        return name
    best, score, _ = rf_process.extractOne(name, options, scorer=rf_fuzz.token_sort_ratio)
    return best if score >= 60 else name


def build_structured_tools(df: pd.DataFrame):
    ALL_CATEGORIES = get_categories(df)
    ALL_INTENTS = get_intents(df)
    state = SelectionState()

    def _normalize_intent(name: Optional[str]) -> Optional[str]:
        return _closest(name, ALL_INTENTS) if name else None

    def _normalize_category(name: Optional[str]) -> Optional[str]:
        return _closest(name, ALL_CATEGORIES) if name else None

    # --------- Selection / Counts / Examples ---------

    @tool
    def select_semantic_intent(intent_names: List[str]) -> str:
        """Select the closest matching intent and store it as active."""
        nonlocal state
        candidates = [_closest(name, ALL_INTENTS) for name in intent_names]
        chosen = candidates[0] if candidates else None
        state.selected_intent = chosen
        if DEBUG:
            print(f"[tools] select_semantic_intent raw={intent_names} -> chosen={chosen}")
        return f"intent_selected: {chosen}"

    @tool
    def select_semantic_category(category_names: List[str]) -> str:
        """Select the closest matching category and store it as active."""
        nonlocal state
        candidates = [_closest(name, ALL_CATEGORIES) for name in category_names]
        chosen = candidates[0] if candidates else None
        state.selected_category = chosen
        if DEBUG:
            print(f"[tools] select_semantic_category raw={category_names} -> chosen={chosen}")
        return f"category_selected: {chosen}"

    @tool
    def count_intent(intent: Optional[str] = None) -> int:
        """Count for an intent; fuzzy-normalize and persist selection."""
        nonlocal state
        name = _normalize_intent(intent or state.selected_intent)
        if not name:
            if DEBUG:
                print("[tools] count_intent -> no intent selected")
            return 0
        state.selected_intent = name
        c = count_by_intent(df, name)
        if DEBUG:
            print(f"[tools] count_intent intent={name} -> {c}")
        return c

    @tool
    def count_category(category: Optional[str] = None) -> int:
        """Count for a category; fuzzy-normalize and persist selection."""
        nonlocal state
        name = _normalize_category(category or state.selected_category)
        if not name:
            if DEBUG:
                print("[tools] count_category -> no category selected")
            return 0
        state.selected_category = name
        c = count_by_category(df, name)
        if DEBUG:
            print(f"[tools] count_category category={name} -> {c}")
        return c

    @tool
    def show_examples(n: int = 5) -> str:
        """Return up to N examples for the current selection.

        Intent takes precedence over category to avoid mismatched slices.
        Fallback order:
          1) intent + category
          2) intent only
          3) category only
        """
        nonlocal state
        intent = state.selected_intent
        category = state.selected_category

        if DEBUG:
            print(f"[tools] show_examples seed intent={intent} category={category} n={n}")

        def _fmt(df_sub: pd.DataFrame):
            out = []
            for _, row in df_sub.iterrows():
                out.append(
                    f"- **{row['intent']} / {row['category']}** — {row['instruction']}\n"
                    f"    ↳ {row['response']}"
                )
            return out

        lines: List[str] = []
        # 1) intent + category
        if intent and category:
            sub = find_examples(df, category=category, intent=intent, n=n)
            lines = _fmt(sub)
            if DEBUG:
                print(f"[tools] show_examples intent+category -> {len(lines)} rows")

        # 2) intent only
        if not lines and intent:
            sub = find_examples(df, intent=intent, n=n)
            lines = _fmt(sub)
            if DEBUG:
                print(f"[tools] show_examples intent-only -> {len(lines)} rows")

        # 3) category only
        if not lines and category:
            sub = find_examples(df, category=category, n=n)
            lines = _fmt(sub)
            if DEBUG:
                print(f"[tools] show_examples category-only -> {len(lines)} rows")

        if not lines and DEBUG:
            print("[tools] show_examples -> (no examples)")

        return "\n".join(lines) if lines else "(no examples)"

    # --------- Lists / Distributions (persist the lists & kind) ---------

    @tool
    def list_categories() -> str:
        """List all available categories (one per line) and persist the list for follow-ups."""
        nonlocal state
        state.last_categories_list = list(ALL_CATEGORIES)
        state.last_list_kind = "category"
        if DEBUG:
            print(f"[tools] list_categories -> {len(state.last_categories_list)} cats (persisted)")
        return "\n".join(f"- {c}" for c in state.last_categories_list) or "(no categories)"

    @tool
    def list_intents() -> str:
        """List all available intents (one per line) and persist the list for follow-ups."""
        nonlocal state
        state.last_intents_list = list(ALL_INTENTS)
        state.last_list_kind = "intent"
        if DEBUG:
            print(f"[tools] list_intents -> {len(state.last_intents_list)} intents (persisted)")
        return "\n".join(f"- {i}" for i in state.last_intents_list) or "(no intents)"

    @tool
    def top_intents(n: int = 10, as_markdown: bool = True, with_bars: bool = True) -> str:
        """Show the top-N most frequent intents with counts, %, and cumulative %.
        Also PERSISTS the ranked top-N intent names for follow-up selections."""
        nonlocal state
        rows, total = _distribution(df, "intent", n)
        # persist only real intents (exclude the synthetic "Other")
        top_names = [name for _, name, _, _, _ in rows if str(name).lower() != "other"]
        state.last_intents_list = top_names
        state.last_list_kind = "intent"
        if DEBUG:
            print(f"[tools] top_intents n={n} -> {len(rows)} rows (total={total}); persisted intents={top_names}")
        return _to_markdown_table(rows, total, header="Intent", with_bars=with_bars) if as_markdown else str(rows)

    @tool
    def top_categories(n: int = 10, as_markdown: bool = True, with_bars: bool = True) -> str:
        """Show the top-N most frequent categories with counts, %, and cumulative %.
        Also PERSISTS the ranked top-N category names for follow-up selections."""
        nonlocal state
        rows, total = _distribution(df, "category", n)
        # persist only real categories (exclude the synthetic "Other")
        top_names = [name for _, name, _, _, _ in rows if str(name).lower() != "other"]
        state.last_categories_list = top_names
        state.last_list_kind = "category"
        if DEBUG:
            print(f"[tools] top_categories n={n} -> {len(rows)} rows (total={total}); persisted categories={top_names}")
        return _to_markdown_table(rows, total, header="Category", with_bars=with_bars) if as_markdown else str(rows)

    @tool
    def count_last_from_previous(
        kind: str = "auto",
        n: int = 2,
        include_breakdown: bool = True,
        as_markdown: bool = True,
    ) -> str:
        """Sum counts for the last N items from the most recently listed/ranked sequence.
        kind: "category" | "intent" | "auto" (auto uses last_list_kind, falling back to whichever list is available)
        n:    how many items from the END to include (clamped to available length)
        Returns a Markdown table (by default) with the breakdown and total.
        """
        nonlocal state

        # decide which list to use
        kind_norm = (kind or "auto").strip().lower()
        cats = state.last_categories_list
        ints = state.last_intents_list
        chosen_kind = None
        seq = []

        if kind_norm in ("category", "intent"):
            chosen_kind = kind_norm
            seq = cats if chosen_kind == "category" else ints
            if not seq:
                # graceful fallback if explicit list is empty
                if chosen_kind == "category" and ints:
                    chosen_kind, seq = "intent", ints
                elif chosen_kind == "intent" and cats:
                    chosen_kind, seq = "category", cats
        else:
            # AUTO: prefer the last_list_kind if populated; else any populated one
            if state.last_list_kind == "category" and cats:
                chosen_kind, seq = "category", cats
            elif state.last_list_kind == "intent" and ints:
                chosen_kind, seq = "intent", ints
            elif cats:
                chosen_kind, seq = "category", cats
            elif ints:
                chosen_kind, seq = "intent", ints

        if DEBUG:
            print(f"[tools] count_last_from_previous kind={kind_norm} -> chosen_kind={chosen_kind} "
                f"len_seq={len(seq)} last_list_kind={state.last_list_kind}")

        if not seq:
            return "(no previous list to count)"

        n = max(1, int(n))
        n = min(n, len(seq))
        window = seq[-n:]  # last N items

        # compute counts
        rows = []
        total = 0
        if chosen_kind == "category":
            for name in window:
                c = count_by_category(df, name)
                rows.append((name, int(c)))
                total += int(c)
        else:
            for name in window:
                c = count_by_intent(df, name)
                rows.append((name, int(c)))
                total += int(c)

        if not include_breakdown:
            return str(total)

        if as_markdown:
            header = "Category" if chosen_kind == "category" else "Intent"
            lines = [f"| {header} | Count |", "|---|---:|"]
            for name, c in rows:
                lines.append(f"| {name} | {c} |")
            lines.append(f"\n**Total (last {n} {header.lower()}s): {total}**")
            return "\n".join(lines)

        # plain text fallback
        breakdown = "\n".join(f"- {name}: {c}" for name, c in rows)
        return f"{breakdown}\n\nTotal: {total}"

    # --------- One-shot “top with examples” helpers ---------

    @tool
    def examples_for_top_categories(k: int = 3, n_per: int = 3) -> str:
        """Return up to n_per examples for each of the top-k categories by frequency and persist that list."""
        nonlocal state
        vc = (
            df["category"].dropna().astype(str).str.strip().value_counts().head(max(1, int(k)))
        )
        cats = list(vc.index)
        state.last_categories_list = cats
        state.last_list_kind = "category"
        lines: List[str] = []
        if DEBUG:
            print(f"[tools] examples_for_top_categories k={k} n_per={n_per} -> cats={cats} (persisted)")
        for cat in cats:
            sub = find_examples(df, category=cat, n=int(n_per))
            lines.append(f"### {cat}")
            if sub.empty:
                lines.append("(no examples)\n")
                continue
            for _, row in sub.iterrows():
                lines.append(
                    f"- **{row['intent']} / {row['category']}** — {row['instruction']}\n"
                    f"    ↳ {row['response']}"
                )
            lines.append("")  # spacer
        return "\n".join(lines).strip() or "(no examples)"

    @tool
    def examples_for_top_intents(k: int = 3, n_per: int = 3) -> str:
        """Return up to n_per examples for each of the top-k intents by frequency and persist that list."""
        nonlocal state
        vc = (
            df["intent"].dropna().astype(str).str.strip().value_counts().head(max(1, int(k)))
        )
        intents = list(vc.index)
        state.last_intents_list = intents
        state.last_list_kind = "intent"
        lines: List[str] = []
        if DEBUG:
            print(f"[tools] examples_for_top_intents k={k} n_per={n_per} -> intents={intents} (persisted)")
        for it in intents:
            sub = find_examples(df, intent=it, n=int(n_per))
            lines.append(f"### {it}")
            if sub.empty:
                lines.append("(no examples)\n")
                continue
            for _, row in sub.iterrows():
                lines.append(
                    f"- **{row['intent']} / {row['category']}** — {row['instruction']}\n"
                    f"    ↳ {row['response']}"
                )
            lines.append("")  # spacer
        return "\n".join(lines).strip() or "(no examples)"

    # --------- Select from the previous list (auto kind + fallback) ---------

    @tool
    def select_from_previous(kind: str = "auto", which: str = "last", n: int = 1) -> str:
        """Select an item from the most recently listed or ranked items.
        kind: "category" | "intent" | "auto"
        which: "last" | "first" | "nth"
        n: index for "nth" (1-based)
        """
        nonlocal state
        kind_norm = (kind or "auto").strip().lower()
        which = (which or "last").strip().lower()
        cats = state.last_categories_list
        ints = state.last_intents_list

        # Decide which sequence to use
        chosen_kind = None
        seq: List[str] = []
        if kind_norm in ("category", "intent"):
            # Respect explicit request, but fall back if empty
            chosen_kind = kind_norm
            seq = cats if kind_norm == "category" else ints
            if not seq:
                # fallback to the other one if available
                if kind_norm == "category" and ints:
                    chosen_kind, seq = "intent", ints
                elif kind_norm == "intent" and cats:
                    chosen_kind, seq = "category", cats
        else:
            # AUTO mode: prefer the last_list_kind if populated; else any populated one
            if state.last_list_kind == "category" and cats:
                chosen_kind, seq = "category", cats
            elif state.last_list_kind == "intent" and ints:
                chosen_kind, seq = "intent", ints
            elif cats:
                chosen_kind, seq = "category", cats
            elif ints:
                chosen_kind, seq = "intent", ints

        if DEBUG:
            print(f"[tools] select_from_previous kind={kind_norm} -> chosen_kind={chosen_kind} len={len(seq)} last_list_kind={state.last_list_kind}")

        if not seq or chosen_kind is None:
            return "none"

        # Index selection
        idx = 0
        if which == "last":
            idx = len(seq) - 1
        elif which == "nth":
            idx = max(0, min(len(seq) - 1, int(n) - 1))
        # else first -> 0

        chosen = seq[idx]
        if chosen_kind == "category":
            state.selected_category = chosen
            return f"category_selected: {chosen}"
        else:
            state.selected_intent = chosen
            return f"intent_selected: {chosen}"

    # --------- Misc tools ---------

    @tool
    def summarize(user_request: str) -> str:
        """Request a free-form summary/analysis (unstructured agent expands it)."""
        if DEBUG:
            print(f"[tools] summarize request={user_request!r}")
        return f"summarize_request: {user_request}"

    @tool
    def sum(a: float, b: float) -> float:
        """Return a + b."""
        if DEBUG:
            print(f"[tools] sum {a} + {b} -> {a + b}")
        return a + b

    @tool
    def finish() -> str:
        """Signal tool-use is finished."""
        if DEBUG:
            print("[tools] finish()")
        return "done"

    # --------- Accessors for graph ---------

    def get_current_selection() -> dict[str, str | None]:
        return {"intent": state.selected_intent, "category": state.selected_category}

    def set_current_selection(intent: Optional[str], category: Optional[str]) -> None:
        """Seed tool state from persisted selections (fuzzy-normalized)."""
        state.selected_intent = _normalize_intent(intent)
        state.selected_category = _normalize_category(category)
        if DEBUG:
            print(f"[tools] set_current_selection -> intent={state.selected_intent} category={state.selected_category}")

    def get_listing_state() -> dict[str, List[str] | Optional[str]]:
        return {
            "last_categories_list": list(state.last_categories_list),
            "last_intents_list": list(state.last_intents_list),
            "last_list_kind": state.last_list_kind,
        }

    def set_listing_state(categories: Optional[List[str]], intents: Optional[List[str]], last_kind: Optional[str] = None) -> None:
        state.last_categories_list = list(categories or [])
        state.last_intents_list = list(intents or [])
        state.last_list_kind = (last_kind or state.last_list_kind)
        if DEBUG:
            print(f"[tools] set_listing_state -> cats={len(state.last_categories_list)} intents={len(state.last_intents_list)} last_kind={state.last_list_kind}")

    tools = [
        # selection / counts / examples
        select_semantic_intent,
        select_semantic_category,
        count_intent,
        count_category,
        show_examples,
        # lists / distributions
        list_categories,
        list_intents,
        top_categories,
        top_intents,
        # one-shot helpers
        examples_for_top_categories,
        examples_for_top_intents,
        # select from previous list
        select_from_previous,
        # NEW: sum counts from last N items
        count_last_from_previous,
        # misc
        summarize,
        sum,
        finish,
    ]

    # expose setters/getters so the graph can persist across turns
    return tools, get_current_selection, set_current_selection, get_listing_state, set_listing_state, ALL_INTENTS, ALL_CATEGORIES
