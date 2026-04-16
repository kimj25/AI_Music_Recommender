"""
Agentic Orchestrator: Conversational Music Recommender

A Claude agent that uses tool use to orchestrate the preference model,
RAG engine, and scoring algorithm in a multi-turn conversation loop.

The agent decides which tools to call based on the user's message,
executes them, and synthesizes results into a natural language response.
"""

import json
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from .preference_model import PreferenceModel
    from .rag import SongRAG
    from .recommender import load_songs, recommend_songs
except ImportError:
    from preference_model import PreferenceModel
    from rag import SongRAG
    from recommender import load_songs, recommend_songs


TOOLS = [
    {
        "name": "extract_music_preferences",
        "description": (
            "Parse the user's natural language description into structured music "
            "preferences (genre, mood, energy, acousticness, valence, tempo). "
            "Use this whenever the user describes a feeling, activity, or vibe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The user's natural language description of what they want to listen to",
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "search_songs_semantic",
        "description": (
            "Search the song catalog using semantic similarity. Best for vague or "
            "contextual queries like 'something for a rainy afternoon' or "
            "'late night drive music' where exact genre matching isn't enough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-form description of the desired music",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of candidates to retrieve (default 8)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Score and rank all catalog songs against structured user preferences. "
            "Returns top-k recommendations with scores and match reasons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_prefs": {
                    "type": "object",
                    "description": (
                        "Structured preferences with fields: genre, mood, energy "
                        "(0-1), likes_acoustic (bool), valence (0-1), tempo_bpm"
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5)",
                },
            },
            "required": ["user_prefs"],
        },
    },
    {
        "name": "get_song_details",
        "description": "Get full metadata for a specific song by title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact song title to look up",
                }
            },
            "required": ["title"],
        },
    },
]

SYSTEM_PROMPT = """You are a friendly and knowledgeable music recommendation assistant. \
Your catalog has 17 songs spanning lofi, pop, rock, ambient, jazz, synthwave, indie pop, \
world, edm, classical, blues, hip hop, folk, and psytrance.

When a user tells you what they want:
1. Call extract_music_preferences to understand their structured preferences
2. Optionally call search_songs_semantic for context-heavy or vague queries
3. Call get_recommendations to rank the catalog against their preferences
4. Present 3-5 recommendations conversationally with a brief reason for each pick

Keep responses warm, concise, and natural. If the user asks for adjustments \
("more upbeat", "something else"), use the tools again with updated parameters."""


class MusicAgent:
    """
    Conversational music recommender powered by Claude with tool use.

    Orchestrates three AI layers:
    - PreferenceModel (Specialized Model): natural language → UserProfile
    - SongRAG (RAG): semantic candidate retrieval
    - recommend_songs (Scoring): rule-based precision ranking

    Model: claude-sonnet-4-6 for strong multi-step reasoning.
    """

    def __init__(self, songs_path: str, descriptions_path: str) -> None:
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"
        self.songs = load_songs(songs_path)
        self.songs_by_title = {s["title"]: s for s in self.songs}
        self.preference_model = PreferenceModel()
        self.rag = SongRAG(self.songs, descriptions_path)

    def _handle_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "extract_music_preferences":
            prefs = self.preference_model.extract(tool_input["description"])
            return json.dumps(prefs)

        if tool_name == "search_songs_semantic":
            n = tool_input.get("n", 8)
            results = self.rag.search(tool_input["query"], n=n)
            return json.dumps(results)

        if tool_name == "get_recommendations":
            prefs = tool_input["user_prefs"]
            k = tool_input.get("k", 5)
            recs = recommend_songs(prefs, self.songs, k=k)
            return json.dumps([
                {
                    "song": song,
                    "score": round(score, 2),
                    "max_score": 11.0,
                    "match_pct": f"{score / 11.0:.0%}",
                    "reasons": reasons,
                }
                for song, score, reasons in recs
            ])

        if tool_name == "get_song_details":
            song = self.songs_by_title.get(tool_input["title"])
            if song:
                return json.dumps(song)
            return json.dumps({"error": f"Song '{tool_input['title']}' not found"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def chat(self, user_message: str, history: list) -> tuple[str, list]:
        """
        Process one conversational turn through the agentic loop.

        Args:
            user_message: The user's latest message
            history: Prior conversation as list of message dicts

        Returns:
            (response_text, updated_history)
        """
        messages = history + [{"role": "user", "content": user_message}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return text, messages

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._handle_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
