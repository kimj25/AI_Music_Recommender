# System Design: AI Music Recommender

## Overview

This project extends a rule-based music recommendation simulation into a full
applied AI system with three AI features:

- **Agentic Workflow** — a Claude agent that orchestrates multi-step recommendation via tool use
- **Retrieval-Augmented Generation (RAG)** — semantic search over song descriptions using ChromaDB
- **Specialized Model** — a domain-tuned Claude classifier that maps natural language to structured preferences

Users interact through a Streamlit chat interface. They describe what they want
in plain English ("something chill for late-night studying"), and the system
discovers their preferences, retrieves candidates, scores them, and explains the
results conversationally.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│               Streamlit Chat UI                  │
│                   (src/app.py)                   │
│  - Multi-turn chat interface                     │
│  - Session state for conversation history       │
│  - @st.cache_resource for agent singleton       │
└─────────────────────┬────────────────────────────┘
                      │ natural language
                      ▼
┌──────────────────────────────────────────────────┐
│           Agentic Orchestrator                   │  ← AI Feature 1
│              (src/agent.py)                      │
│  - Claude claude-haiku-4-5-20251001 with tool use       │
│  - Multi-turn conversation loop                  │
│  - Routes to tools based on user intent          │
│  - Synthesizes results into natural language     │
└────────┬──────────────┬──────────────┬───────────┘
         │              │              │
         ▼              ▼              ▼
┌────────────┐  ┌───────────────┐  ┌────────────────┐
│Specialized │  │  RAG Engine   │  │    Scoring     │
│  Model     │  │  (src/rag.py) │  │  Algorithm     │
│(src/prefer-│  │               │  │(src/recommender│
│ence_model  │  │  - ChromaDB   │  │    .py)        │
│   .py)     │  │  - Sentence   │  │                │
│            │  │  Transformers │  │  - score_song()│
│- claude-   │  │  embeddings   │  │  - recommend_  │
│  haiku for │  │  - semantic   │  │    songs()     │
│  fast      │  │    search     │  │  (existing,    │
│  classifi- │  │  over song    │  │   unchanged)   │
│  cation    │  │  descriptions │  │                │
│- few-shot  │  │               │  │                │
│  examples  │  │               │  │                │
└────────────┘  └───────────────┘  └────────────────┘
  AI Feature 3     AI Feature 2       Existing logic
```

---

## Data Flow

```
User: "I want something mellow for a rainy afternoon"
  │
  ▼
[Agent] decides to call tools
  │
  ├─► [extract_music_preferences("mellow for a rainy afternoon")]
  │       └─► PreferenceModel (Haiku + few-shot) → {genre:"blues", mood:"melancholic", energy:0.4, ...}
  │
  ├─► [search_songs_semantic("mellow rainy afternoon")]
  │       └─► SongRAG → ChromaDB query → [Library Rain, Coffee Shop Stories, Broken Streetlight, ...]
  │
  └─► [get_recommendations({genre:"blues", mood:"melancholic", energy:0.4, ...}, k=5)]
          └─► recommend_songs() → scored + ranked results
  │
  ▼
[Agent] synthesizes tool results into conversational response
  │
  ▼
"Here are some picks for your rainy afternoon — Broken Streetlight has a soulful
blues feel that matches perfectly, and Library Rain gives you that quiet, introspective
vibe..."
```

---

## Components

### 1. Specialized Model (`src/preference_model.py`)

**Purpose:** Convert free-form natural language into structured `UserProfile` dicts
compatible with the existing scoring algorithm.

**Why a specialized model:**
- Generic LLMs can parse preferences, but produce inconsistent JSON schemas
- Domain-specific few-shot examples anchor outputs to the exact genre/mood vocabulary
  used in the song catalog
- Uses `claude-haiku-4-5` for fast, cheap classification (no heavy reasoning needed)

**Key design decisions:**
- System prompt enumerates all valid genres and moods from the catalog
- 3 few-shot examples cover the energy spectrum (chill → gym → sad)
- Returns JSON directly — no parsing of verbose text
- Used as a tool by the agent (not called standalone in production)

**Schema output:**
```json
{
  "genre": "lofi",
  "mood": "focused",
  "energy": 0.35,
  "likes_acoustic": true,
  "valence": 0.55,
  "tempo_bpm": 78.0
}
```

---

### 2. RAG Engine (`src/rag.py`)

**Purpose:** Enable semantic search over the song catalog using natural language
queries that structured scoring alone can't handle.

**Why RAG:**
- The scoring algorithm requires exact genre/mood labels — it can't handle "something
  for a late-night drive" or "music that feels like autumn"
- Semantic embeddings capture meaning that keyword matching misses
- Acts as a pre-filter to surface candidates the scoring algorithm then re-ranks

**Architecture:**
- `data/song_descriptions.json` — curated natural language descriptions per song
- ChromaDB in-memory vector store (no external server needed)
- Default sentence-transformers embedding (`all-MiniLM-L6-v2`)
- Returns top-N songs by semantic similarity for agent to re-rank

**Hybrid retrieval pattern:**
```
RAG (semantic recall) → scoring algorithm (precision ranking)
```

---

### 3. Agentic Orchestrator (`src/agent.py`)

**Purpose:** Multi-turn conversational agent that intelligently routes between
tools to answer any music request.

**Tools available:**

| Tool | When used |
|------|-----------|
| `extract_music_preferences` | User describes mood/activity in natural language |
| `search_songs_semantic` | Context-based queries ("something for rainy days") |
| `get_recommendations` | Score and rank songs against structured preferences |
| `get_song_details` | User asks about a specific song |

**Agentic loop:**
1. Receive user message + conversation history
2. Send to Claude with tools
3. If `stop_reason == "tool_use"`: execute tools, append results, loop
4. If `stop_reason == "end_turn"`: return final text response

**Model:** `claude-haiku-4-5-20251001` (switched from Sonnet to stay under $5 API budget)

---

### 4. Streamlit Chat UI (`src/app.py`)

**Purpose:** Web-based conversational interface.

**Key features:**
- `st.chat_message` for proper chat layout
- `st.session_state` preserves conversation history across turns
- `@st.cache_resource` ensures the agent (and ChromaDB index) is created once
- Spinner feedback during agent processing

---

## File Structure

```
music_recommender_simulation/
├── data/
│   ├── songs.csv                    # 40-song catalog (existing)
│   └── song_descriptions.json       # NEW: rich text per song for RAG
├── src/
│   ├── recommender.py               # UNCHANGED: scoring algorithm
│   ├── main.py                      # UNCHANGED: CLI demo
│   ├── preference_model.py          # NEW: Specialized Model
│   ├── rag.py                       # NEW: RAG Engine
│   ├── agent.py                     # NEW: Agentic Orchestrator
│   └── app.py                       # NEW: Streamlit UI
├── tests/
│   ├── test_recommender.py          # EXISTING
│   ├── test_preference_model.py     # NEW
│   ├── test_rag.py                  # NEW
│   └── test_agent.py                # NEW
├── DESIGN.md                        # NEW: this file
├── model_card.md
├── README.md
└── requirements.txt                 # UPDATED
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (agent + specialized model) | Anthropic Claude (claude-haiku-4-5-20251001) |
| Vector store | ChromaDB (in-memory) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (via ChromaDB default) |
| UI | Streamlit |
| Data | pandas, CSV |
| Testing | pytest with mocked API calls |
| Secrets | python-dotenv (.env file) |

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run Streamlit chat UI
streamlit run src/app.py

# Run CLI demo (original)
python -m src.main

# Run tests
pytest tests/
```

---

## Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Haiku for all Claude calls | Fast + cheap for both classification and orchestration; keeps total spend under $5 |
| ChromaDB in-memory | No infrastructure setup; fine for 40-song catalog |
| RAG + scoring hybrid | RAG has high recall, scoring has precision — combination is better than either alone |
| Tool use over direct calls | Agent decides when/what to call based on context, handles follow-ups naturally |
| Few-shot over fine-tuning | Catalog is too small (40 songs) to justify fine-tuning; few-shot with fixed vocabulary is sufficient |
| Existing scoring untouched | Rule-based scoring is interpretable and correct — no need to replace it |
