# Bitext Data Analyst Agent (LangGraph + Nebius)

Implements a Data Analyst Agent using LangGraph with:
- Router node for structured / unstructured / out-of-scope / memory
- Two sub-agents (structured & unstructured), each with its own tools
- Follow-up queries with LangGraph checkpoints + thread IDs (works across tabs/reloads)
- Summarized memory with a summary node
- Streamlit chat app with Session ID input

## Requirements
- Python 3.11 (recommended)
- Nebius AI Studio API key (`NEBIUS_API_KEY`)
- Optional: `NEBIUS_MODEL` (default: `Qwen/Qwen2.5-32B-Instruct`)

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```
NEBIUS_API_KEY=***
NEBIUS_MODEL=Qwen/Qwen2.5-32B-Instruct
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
```

## Dataset
By default loads from Hugging Face: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`.
Alternatively place `./data/bitext.csv` with columns: flags,instruction,category,intent,response.
A tiny fallback sample is bundled so the app always runs.

## Run
```bash
streamlit run app.py
```

Checkpoints are stored at `./checkpoints/bitext.sqlite` and keyed by your Session ID (thread_id).
