import os
import uuid
import streamlit as st

from core.graph_factory import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver  # app owns the saver


st.set_page_config(page_title="Bitext Data Analyst Agent", page_icon="🧠", layout="wide")
st.title("🧠 Bitext Data Analyst Agent")

# --- Internal thread id separate from widget key ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

st.sidebar.header("Session")

# Text input uses a DIFFERENT key than the state var we mutate
sid = st.sidebar.text_input(
    "Session ID (thread_id)",
    value=st.session_state.thread_id,
    key="session_id_input",
    help="Use the same ID across tabs to share memory and follow-ups.",
)

# If user manually edited the value, sync it to our internal thread_id and rerun
if sid != st.session_state.thread_id:
    st.session_state.thread_id = sid.strip() or st.session_state.thread_id
    # reset only the UI history; LangGraph thread persists in DB
    st.session_state.history = []
    st.rerun()

# --- Checkpointer: create/enter once and keep for the session ---
if "checkpointer" not in st.session_state:
    os.makedirs("checkpoints", exist_ok=True)
    cm = SqliteSaver.from_conn_string("checkpoints/bitext.sqlite")  # context manager
    st.session_state._cp_ctx = cm                                   # keep reference so it isn’t GC’d
    st.session_state.checkpointer = cm.__enter__()                   # enter context and hold saver

# --- Build graph once with the provided saver ---
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(checkpointer=st.session_state.checkpointer)
graph = st.session_state.graph

st.sidebar.caption("Persistence: LangGraph SqliteSaver @ ./checkpoints/bitext.sqlite")

# --- Simple chat history for UI (LangGraph state lives in DB) ---
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# --- Helper: run a prompt with state hydration from checkpointer ---
def run_prompt(prompt_text: str):
    # UI echo
    st.session_state.history.append(("user", prompt_text))
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Thread config
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Hydrate prior state from checkpointer so tools reuse last selection + memory
    snapshot = graph.get_state(config)
    persisted = snapshot.values if snapshot is not None else {}

    state_input = {
        "user_message": prompt_text,
        "selected_intent": persisted.get("selected_intent"),
        "selected_category": persisted.get("selected_category"),
        "summary_memory": persisted.get("summary_memory") or {},
        # hydrate the last lists + kind so “last one on your list” works
        "last_categories_list": persisted.get("last_categories_list") or (persisted.get("summary_memory") or {}).get("last_categories_list"),
        "last_intents_list": persisted.get("last_intents_list") or (persisted.get("summary_memory") or {}).get("last_intents_list"),
        "last_list_kind": persisted.get("last_list_kind") or (persisted.get("summary_memory") or {}).get("last_list_kind"),
    }

    result = graph.invoke(state_input, config=config)
    answer = result.get("final_response", "(no answer)")

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))

# --- Chat input ---
prompt = st.chat_input("Ask about intents, categories, counts or summaries…")
if prompt:
    run_prompt(prompt)

# --- Sidebar quick test buttons that actually invoke the graph ---
st.sidebar.divider()
st.sidebar.subheader("Quick tests")

if st.sidebar.button("How many refund requests did we get?"):
    run_prompt("how many refund requests did we get?")
    st.stop()  # avoid duplicate rendering

if st.sidebar.button("Show examples"):
    run_prompt("show examples")
    st.stop()

st.sidebar.divider()

# Reset without touching the widget key; just change our internal thread id
if st.sidebar.button("Reset current session (new thread_id)"):
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.history = []
    st.toast(f"New Session ID created: {st.session_state.thread_id}. This tab now uses a fresh thread.")
    st.rerun()

st.markdown(
    """
**Tips**
- Follow-ups work: _“Show me more examples”, “What is the total count of the last two intents?”_
- Ask: _“What do you remember about me?”_ to query summarized memory.
- Enable debug logs by putting `DEBUG_AGENT=1` in your `.env` and restart.
"""
)
