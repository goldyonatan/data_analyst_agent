# Bitext Data Analyst Agent (LangGraph + Streamlit)

A small, reproducible **data‑analyst agent** built with **LangGraph** and a **Streamlit** UI.  
It routes queries (structured / unstructured / out‑of‑scope / memory), uses **LangGraph checkpoints** for follow‑ups, and keeps a short **summary memory** between turns. It now supports either **OpenAI‑compatible** (default) or **Nebius** backends via a simple `PROVIDER` switch.

---

## Features
- **Router node** that dispatches to structured / unstructured sub‑agents or memory / OoS paths
- **Two sub‑agents** (structured & unstructured), each with its own tools
- **Follow‑ups that work** across tabs/reloads using **LangGraph checkpoints + thread IDs**
- **Summary memory** so you can ask “what do you remember about me?”
- **Streamlit chat UI** with a **Session ID** (a.k.a. `thread_id`) field in the sidebar
- **Provider toggle**: `PROVIDER=openai` (default) or `PROVIDER=nebius`

---

## Quick Start (one‑command)
If you already have Python 3.11+ and an OpenAI API key:
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt \
&& cp .env.example .env && sed -i.bak 's/OPENAI_API_KEY=.*/OPENAI_API_KEY=replace_me/' .env \
&& streamlit run app.py
```
> On Windows (PowerShell), replace `sed -i.bak ...` with manually editing `.env` to paste your key.

You’ll see `Provider: openai • Persistence: ./checkpoints/bitext.sqlite` in the sidebar, and you can type “hello” or click the **Quick tests** buttons.

---

## Requirements
- **Python** 3.11 (recommended)
- **One** of the following:
  - `OPENAI_API_KEY` (default provider), optionally `OPENAI_MODEL` (defaults to `gpt-4o-mini`)
  - **or** `NEBIUS_API_KEY` (and optional `NEBIUS_MODEL`, defaults to `Qwen/Qwen2.5-32B-Instruct`)
- Internet access if you want to pull the default HF dataset (the app also supports a local CSV fallback)

### Python dependencies
See `requirements.txt`. At minimum, you’ll need:
- `streamlit`
- `langgraph`
- `langchain` and `langchain-openai` (for the OpenAI provider)
- `python-dotenv`
- `datasets`
- `pandas`

> Pin versions as you see fit (e.g., `streamlit>=1.34`, `langgraph>=0.2`, `langchain>=0.2`, `langchain-openai>=0.1`, `python-dotenv>=1.0`, `datasets>=2.19`, `pandas>=2.2`).

---

## Installation
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Environment
Fill a `.env` file at repo root. Use the example below or copy from `.env.example`.

```bash
# Provider selection
PROVIDER=openai            # or: nebius

# OpenAI-compatible (default)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini   # optional

# Nebius (OpenAI-compatible endpoint)
NEBIUS_API_KEY=
NEBIUS_MODEL=Qwen/Qwen2.5-32B-Instruct   # optional

# Optional LangSmith tracing (disabled by default)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=

# Optional: enable extra debug logs in the app
DEBUG_AGENT=0
```

Create `.env` quickly:
```bash
cp .env.example .env
# then open .env and paste your key(s)
```

---

## Dataset
By default, the app can read a public Hugging Face dataset:
```
bitext/Bitext-customer-support-llm-chatbot-training-dataset
```
Alternatively, provide a local CSV at `./data/bitext.csv` with columns:
```
flags,instruction,category,intent,response
```
If HF download fails and no CSV is present, the code path can fall back to a tiny in-memory sample so the app still boots.

> Tip: If you rely on the local file, ensure the `data/` folder exists. The app doesn’t write to it by default.

---

## Run
```bash
streamlit run app.py
```
- A `checkpoints/` folder will be created automatically (if it doesn’t exist).  
- Checkpoints are stored at `./checkpoints/bitext.sqlite` and keyed by your **Session ID** (`thread_id`).  
- You can set or copy the `thread_id` from the sidebar to share state across tabs.

**Quick tests** (sidebar):
- “How many refund requests did we get?”
- “Show examples”

**Example free-form prompts**:
- “Summarize the last two intents.”
- “What do you remember about me?”

---

## How PROVIDER selection works
`app.py` reads `PROVIDER` from the environment:
- `openai` (default): the app constructs an LLM with `langchain_openai.ChatOpenAI` using `OPENAI_API_KEY` and `OPENAI_MODEL` (default: `gpt-4o-mini`).
- `nebius`: the app hands off LLM creation to `core/graph_factory.py`. Set `NEBIUS_API_KEY` (and optionally `NEBIUS_MODEL`).

You’ll see the active provider in the Streamlit sidebar caption.

---

## Project structure (key files)
```
app.py
core/
  __init__.py
  graph_factory.py
checkpoints/            # auto-created on first run (do not commit the sqlite file)
data/
  bitext.csv            # optional local dataset (not committed)
.env.example
requirements.txt
README.md
```

---

## Troubleshooting
- **`OPENAI_API_KEY is required when PROVIDER=openai`**  
  You picked the default provider but didn’t set a key. Copy `.env.example` → `.env` and paste your key.
- **`ModuleNotFoundError: core.graph_factory`**  
  Ensure `core/__init__.py` exists and you’re running from repo root (`streamlit run app.py`).
- **`sqlite` / checkpoints errors**  
  The app now ensures `./checkpoints` exists before connecting. If you manually remove it during a run, restart the app.
- **Dataset errors / offline runs**  
  Add a small `data/bitext.csv` locally or rely on the built-in tiny in-memory sample.
- **Wrong Python version**  
  Use Python 3.11 (recommended). Other versions may work but are not tested here.

---

## FAQ
**Q: Can I run without Nebius?**  
A: Yes. The default `PROVIDER=openai` path only needs `OPENAI_API_KEY`.

**Q: Where is conversation state stored?**  
A: In `./checkpoints/bitext.sqlite`. It’s keyed by `thread_id` (the Session ID in the sidebar).

**Q: Can I switch tabs and keep the same memory?**  
A: Yes. Copy the Session ID and reuse it in another tab/instance.
