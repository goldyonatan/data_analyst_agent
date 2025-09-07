from __future__ import annotations
import os
import pandas as pd
from typing import Optional

FALLBACK_ROWS = [
    {"flags": "", "instruction": "I want a refund for my order", "category": "REFUNDS", "intent": "get_refund", "response": "I can help with that. Please provide your order number to begin the refund process."},
    {"flags": "", "instruction": "Where is my order?", "category": "ORDER_STATUS", "intent": "track_order", "response": "Let me check your order status. Could you share your order ID?"},
    {"flags": "", "instruction": "I need to change my shipping address", "category": "ACCOUNT", "intent": "update_profile", "response": "Sure, I can help you update your shipping address."},
]

REQUIRED_COLUMNS = ["flags", "instruction", "category", "intent", "response"]

def _try_load_from_csv(csv_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def _try_load_from_hf() -> Optional[pd.DataFrame]:
    try:
        from datasets import load_dataset
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        df = ds.to_pandas()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            rename_map = {}
            for c in REQUIRED_COLUMNS:
                if c not in df.columns:
                    low = [x for x in df.columns if x.lower() == c.lower()]
                    if low:
                        rename_map[low[0]] = c
            if rename_map:
                df = df.rename(columns=rename_map)
            missing2 = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing2:
                raise ValueError(f"HF dataset missing required columns after rename: {missing2}")
        return df
    except Exception as e:
        print("[dataset] HuggingFace load failed:", e)
        return None

def load_bitext_dataframe() -> pd.DataFrame:
    csv_env = os.getenv("BITEXT_CSV", "data/bitext.csv")
    df = _try_load_from_csv(csv_env)
    if df is not None:
        return df
    df = _try_load_from_hf()
    if df is not None:
        return df
    print("[dataset] Using tiny baked-in sample.")
    return pd.DataFrame(FALLBACK_ROWS)

def get_categories(df: pd.DataFrame):
    return sorted(list({str(x).strip() for x in df["category"].dropna().unique()}))

def get_intents(df: pd.DataFrame):
    return sorted(list({str(x).strip() for x in df["intent"].dropna().unique()}))

def count_by_category(df: pd.DataFrame, category: str) -> int:
    return int((df["category"].str.strip().str.lower() == category.strip().lower()).sum())

def count_by_intent(df: pd.DataFrame, intent: str) -> int:
    return int((df["intent"].str.strip().str.lower() == intent.strip().lower()).sum())

def find_examples(df: pd.DataFrame, *, category: str | None = None, intent: str | None = None, n: int = 5) -> pd.DataFrame:
    sub = df
    if category:
        sub = sub[sub["category"].str.strip().str.lower() == category.strip().lower()]
    if intent:
        sub = sub[sub["intent"].str.strip().str.lower() == intent.strip().lower()]
    return sub.head(n)[["instruction", "category", "intent", "response"]].copy()
