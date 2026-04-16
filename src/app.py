"""
Streamlit Chat UI for the AI Music Recommender.

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on the path when running via `streamlit run src/app.py`
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from project root explicitly — Streamlit's CWD can vary
load_dotenv(ROOT / ".env")

import streamlit as st
from src.agent import MusicAgent

SONGS_PATH = "data/songs.csv"
DESCRIPTIONS_PATH = "data/song_descriptions.json"

st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 AI Music Recommender")
st.caption(
    "Tell me what you're in the mood for — an activity, a feeling, or a vibe — "
    "and I'll find the right songs from the catalog."
)


@st.cache_resource(show_spinner="Loading music catalog...")
def get_agent() -> MusicAgent:
    return MusicAgent(SONGS_PATH, DESCRIPTIONS_PATH)


agent = get_agent()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: clear button only
with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("What kind of music are you feeling?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding songs..."):
            response, updated_history = agent.chat(prompt, st.session_state.history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history = updated_history
    st.rerun()
