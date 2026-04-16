# 🎵 AI Music Recommender Simulation

## Project Summary

MusicMate Finder 2.0 is a content-based music recommender extended into a full applied AI system. It scores songs against a user's taste profile — genre, mood, energy, valence, tempo, and acoustic preference — and returns the top matches with explanations. Users interact through a conversational Streamlit chat interface powered by a Claude agent.

**AI Features:**
- **Agentic Workflow** — Claude agent with tool use that orchestrates multi-step recommendations through natural language conversation
- **Retrieval-Augmented Generation (RAG)** — semantic search over song descriptions using ChromaDB, enabling queries like "music for a rainy afternoon"
- **Specialized Model** — domain-tuned Claude classifier that maps free-form text to structured user preferences

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│               Streamlit Chat UI                  │
│                   (src/app.py)                   │
└─────────────────────┬────────────────────────────┘
                      │ natural language
                      ▼
┌──────────────────────────────────────────────────┐
│           Agentic Orchestrator                   │
│              (src/agent.py)                      │
│         Claude Haiku + tool use                  │
└────────┬──────────────┬──────────────┬───────────┘
         │              │              │
┌────────▼────┐  ┌───────▼───────┐  ┌──▼─────────────┐
│ Specialized │  │  RAG Engine   │  │    Scoring     │
│   Model     │  │  (src/rag.py) │  │  Algorithm     │
│(src/prefer- │  │  ChromaDB +   │  │(src/recommender│
│ence_model   │  │  embeddings   │  │    .py)        │
│   .py)      │  │               │  │                │
└─────────────┘  └───────────────┘  └────────────────┘
```

For full design details see [DESIGN.md](assets/DESIGN.md).

---

## How the Scoring Works

Each `Song` has: `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`.

Every song is scored against the user profile and top results are returned with a breakdown of what drove each score.

**Categorical matches (binary)**

| Feature | Points | Notes |
|---|---|---|
| Genre match | +3.0 | Strongest taste signal |
| Mood match | +2.0 | Situational and intentional |
| Acoustic preference match | +1.0 | When `likes_acoustic` is true and `acousticness > 0.6` |
| Acoustic preference mismatch | -1.0 | When `likes_acoustic` is false and `acousticness > 0.6` |

**Continuous similarity (partial credit)**

| Feature | Max Points | Formula |
|---|---|---|
| Energy | 2.0 | `2.0 × (1 - │target_energy - song.energy│)` |
| Valence | 1.5 | `1.5 × (1 - │target_valence - song.valence│)` |
| Tempo | 1.0 | `1.0 × (1 - │target_tempo - song.tempo│ / 100)` |
| Danceability | 0.5 | Always-on bonus |

**Maximum possible score: 11.0 points**

**Data flow diagram:** [flowchart.md](flowchart.md)

---

## Getting Started

### Prerequisites

- Python 3.12+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac / Linux
   venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your API key in a `.env` file at the project root:

   ```
   ANTHROPIC_API_KEY=sk-ant-...your-key...
   ```

---

## Running the App

**Streamlit chat UI (AI system):**
```bash
venv/bin/streamlit run src/app.py
```

**Original CLI demo:**
```bash
python -m src.main
```

**Tests:**
```bash
venv/bin/python -m pytest tests/
```

---

## Demo

**User Preference Profile**
![User Preference Profile](assets/user_prefs.png)

**Top 5 Recommendations**
![Top 5 Recommendations Part 1](assets/top5_1.png)
![Top 5 Recommendations Part 2](assets/top5_2.png)

---

## Limitations and Risks

- Catalog only has 40 songs
- Genre and mood are exact string matches — similar but differently labeled songs score as misses
- Danceability is always rewarded regardless of user preference, creating an unintentional bias toward EDM and hip hop

---

## Reflection

[**Model Card**](model_card.md)

Building this system made it clear how much a recommender's behavior is shaped by its scoring weights — a small change like raising the genre weight from 2.0 to 3.0 significantly narrows what gets recommended. It also showed how bias can be structural rather than intentional: the danceability bonus favors certain genres by default, not because of any deliberate choice but simply because the feature was added without tying it to user preference.

Extending the project into a full AI system showed how each layer addresses a different gap: the specialized model bridges natural language to structured data, RAG handles vague semantic queries that exact matching can't, and the agent ties everything together into a coherent conversation. Working with AI tools throughout reinforced that human review remains essential — test cases and careful inspection catch things that look correct but quietly test the wrong thing.
