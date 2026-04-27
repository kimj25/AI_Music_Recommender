# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  : **MusicMate Finder 2.0**

---

## 2. Intended Use  

MusicMate Finder 2.0 is a conversational AI music recommender. Users describe what they want to listen to in natural language — a mood, activity, or vibe — and the system returns personalized song recommendations with score explanations. It is built on top of the original rule-based scoring engine from 1.0, extended with a Claude Haiku agent, RAG, and a specialized preference model.

---

## 3. How the Model Works  

User input flows through four layers:

1. **Streamlit Chat UI** — accepts natural language from the user
2. **Agentic Orchestrator** (Claude Haiku + tool use) — decides which tools to call and assembles the final response
3. **Specialized Model** (Claude Haiku, few-shot) — maps free-form text to a structured `UserProfile` (genre, mood, energy, valence, tempo, acousticness)
4. **RAG Engine** (ChromaDB + sentence-transformers) — retrieves semantically relevant songs from text descriptions for vague queries like "music for a rainy afternoon"

The structured profile is then passed to the original **scoring algorithm**, which ranks all catalog songs by weighted feature similarity and returns the top matches with per-song score breakdowns.

---

## 4. Data  

The catalog contains **40 songs across 27 genres** — including pop, lofi, rock, jazz, classical, hip hop, EDM, R&B, country, K-pop, Latin, folk, and ambient — with deep coverage in several genres (country: 4, R&B: 5, K-pop: 4, lofi: 3). Each song has a rich text description stored in `data/song_descriptions.json` used as the RAG corpus.

---

## 5. Strengths  

- Handles vague, natural language queries that exact matching can't — RAG bridges the gap between how users describe music and how it's categorized
- The specialized model removes the burden of structured input from the user; they can say "something upbeat for the gym" and get a valid preference profile
- Per-song score explanations make recommendations transparent and interpretable
- Works best for users with a clear preference profile, but now also handles open-ended requests through semantic search

---

## 6. Limitations and Bias 

- Genre and mood are still exact string matches in the scoring layer — a song close in sound but labeled differently will score as a miss
- Danceability is always rewarded regardless of user preference, creating an unintentional bias toward EDM and hip hop
- The catalog is small (40 songs); genre diversity is limited and recommendations can feel repetitive within a genre
- The specialized model relies on few-shot prompting, so unusual or highly specific inputs may produce imprecise preference profiles

---

## 7. Evaluation  

The scoring algorithm was tested against three user profiles — a calm lofi listener, a high-energy dance listener, and a deep rock listener. Top-ranked songs were checked for genre/mood alignment and score explanations were verified to correctly reflect what drove each result.

The AI layers were evaluated with unit tests (`tests/test_preference_model.py`, `tests/test_rag.py`, `tests/test_agent.py`). The most notable finding was that the specialized model occasionally over-weighted genre when input was ambiguous, and that RAG results improved significantly with longer, more descriptive song descriptions in the corpus.

---

## 8. Future Work  

- Add mismatch penalties so songs that actively conflict with user preferences score lower, not just zero
- Add a diversity constraint to prevent the same genre from filling multiple top-5 slots
- Allow users to adjust feature weights so they can control how much genre vs. energy matters
- Expand the catalog beyond 40 songs to reduce repetition and improve discovery
- Fine-tune the preference extraction prompt to handle edge cases like mixed-genre or cross-mood requests

---

## 9. Personal Reflection  

Extending MusicMate Finder into a full AI system made it clear how each layer solves a different problem: the specialized model handles the gap between natural language and structured data, RAG handles semantic intent that exact matching misses, and the agent ties everything together into a coherent conversation. The original scoring logic stayed completely unchanged — the new layers wrap around it rather than replace it, which showed how modular design makes a system easier to extend.

Working with AI tools throughout reinforced that human review remains essential. Test cases and careful inspection catch things that look correct but quietly test the wrong thing. The biggest takeaway is that building with AI accelerates development, but the judgment about what to build, how to test it, and what counts as correct still has to come from the developer.
