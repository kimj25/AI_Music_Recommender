"""
Specialized Model: Music Preference Extractor

A domain-tuned Claude model that converts natural language descriptions
into structured UserProfile dicts compatible with the scoring algorithm.

Uses claude-haiku for fast, low-cost classification with few-shot examples
anchored to the exact genre/mood vocabulary of the song catalog.
"""

import json
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GENRES = [
    "lofi", "pop", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "world", "edm", "classical", "blues",
    "hip hop", "folk", "psytrance",
]

MOODS = [
    "chill", "intense", "happy", "relaxed", "focused", "moody",
    "adventurous", "energetic", "calm", "melancholic", "confident",
    "peaceful", "psychedelic",
]

SYSTEM_PROMPT = f"""You are a music taste analyst. Given a user's natural language description \
of what they want to listen to, extract their music preferences as structured JSON.

Available genres: {", ".join(GENRES)}
Available moods: {", ".join(MOODS)}

Always return valid JSON with exactly these fields:
{{
  "genre": "<one of the available genres>",
  "mood": "<one of the available moods>",
  "energy": <float 0.0-1.0>,
  "likes_acoustic": <true or false>,
  "valence": <float 0.0-1.0>,
  "tempo_bpm": <float, typical range 60-180>
}}

Energy guide: 0.0-0.3 very calm, 0.3-0.5 relaxed, 0.5-0.7 moderate, 0.7-0.85 energetic, 0.85-1.0 intense
Valence guide: 0.0-0.3 dark/sad, 0.3-0.5 bittersweet, 0.5-0.7 positive, 0.7-1.0 uplifting/happy
Return only the JSON object. No explanation, no markdown."""

# Few-shot examples covering low / high / sad energy profiles
FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": "I want something chill for studying late at night",
    },
    {
        "role": "assistant",
        "content": '{"genre": "lofi", "mood": "focused", "energy": 0.35, "likes_acoustic": true, "valence": 0.55, "tempo_bpm": 78}',
    },
    {
        "role": "user",
        "content": "I need something to pump me up for the gym",
    },
    {
        "role": "assistant",
        "content": '{"genre": "edm", "mood": "energetic", "energy": 0.90, "likes_acoustic": false, "valence": 0.78, "tempo_bpm": 138}',
    },
    {
        "role": "user",
        "content": "Something sad and reflective for a rainy day",
    },
    {
        "role": "assistant",
        "content": '{"genre": "blues", "mood": "melancholic", "energy": 0.40, "likes_acoustic": true, "valence": 0.28, "tempo_bpm": 82}',
    },
    {
        "role": "user",
        "content": "Background music for a cozy Sunday morning with coffee",
    },
    {
        "role": "assistant",
        "content": '{"genre": "jazz", "mood": "relaxed", "energy": 0.35, "likes_acoustic": true, "valence": 0.70, "tempo_bpm": 88}',
    },
]


class PreferenceModel:
    """
    Specialized Claude model for extracting structured music preferences
    from natural language. Uses domain-specific few-shot examples to
    reliably map free-form text to UserProfile schema.

    Model: claude-haiku-4-5 — fast and cheap for single-field classification.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"

    def extract(self, user_description: str) -> dict:
        """
        Convert a natural language music request into a structured UserProfile dict.

        Args:
            user_description: Free-form text, e.g. "something mellow for a rainy afternoon"

        Returns:
            Dict with keys: genre, mood, energy, likes_acoustic, valence, tempo_bpm
        """
        messages = FEW_SHOT + [{"role": "user", "content": user_description}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        raw = response.content[0].text.strip()
        return json.loads(raw)
