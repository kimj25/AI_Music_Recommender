"""
Tests for the Specialized Model (PreferenceModel).

All tests mock the Anthropic API to avoid real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


def _mock_response(payload: dict) -> MagicMock:
    block = MagicMock()
    block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [block]
    return response


REQUIRED_FIELDS = ["genre", "mood", "energy", "likes_acoustic", "valence", "tempo_bpm"]

VALID_GENRES = [
    "lofi", "pop", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "world", "edm", "classical", "blues",
    "hip hop", "folk", "psytrance",
]

VALID_MOODS = [
    "chill", "intense", "happy", "relaxed", "focused", "moody",
    "adventurous", "energetic", "calm", "melancholic", "confident",
    "peaceful", "psychedelic",
]


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_returns_all_required_fields(mock_cls):
    from src.preference_model import PreferenceModel

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "lofi", "mood": "focused", "energy": 0.35,
        "likes_acoustic": True, "valence": 0.55, "tempo_bpm": 78.0,
    })

    result = PreferenceModel().extract("late-night study session")

    for field in REQUIRED_FIELDS:
        assert field in result, f"Missing field: {field}"


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_energy_is_normalized(mock_cls):
    from src.preference_model import PreferenceModel

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "edm", "mood": "energetic", "energy": 0.92,
        "likes_acoustic": False, "valence": 0.80, "tempo_bpm": 138.0,
    })

    result = PreferenceModel().extract("pump me up for the gym")

    assert 0.0 <= result["energy"] <= 1.0


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_likes_acoustic_is_bool(mock_cls):
    from src.preference_model import PreferenceModel

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "jazz", "mood": "relaxed", "energy": 0.38,
        "likes_acoustic": True, "valence": 0.68, "tempo_bpm": 88.0,
    })

    result = PreferenceModel().extract("coffee shop sunday morning")

    assert isinstance(result["likes_acoustic"], bool)


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_high_energy_query_maps_correctly(mock_cls):
    from src.preference_model import PreferenceModel

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "edm", "mood": "energetic", "energy": 0.90,
        "likes_acoustic": False, "valence": 0.78, "tempo_bpm": 140.0,
    })

    result = PreferenceModel().extract("I need maximum energy for my workout")

    assert result["energy"] >= 0.7
    assert result["likes_acoustic"] is False


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_sad_query_maps_low_valence(mock_cls):
    from src.preference_model import PreferenceModel

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "blues", "mood": "melancholic", "energy": 0.40,
        "likes_acoustic": True, "valence": 0.28, "tempo_bpm": 82.0,
    })

    result = PreferenceModel().extract("something sad and reflective")

    assert result["valence"] <= 0.4


@patch("src.preference_model.anthropic.Anthropic")
def test_extract_output_is_compatible_with_score_song(mock_cls):
    """PreferenceModel output should work directly as user_prefs in score_song."""
    from src.preference_model import PreferenceModel
    from src.recommender import score_song

    mock_cls.return_value.messages.create.return_value = _mock_response({
        "genre": "lofi", "mood": "chill", "energy": 0.42,
        "likes_acoustic": True, "valence": 0.56, "tempo_bpm": 78.0,
    })

    prefs = PreferenceModel().extract("something chill")

    song = {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill", "energy": 0.42,
        "tempo_bpm": 78.0, "valence": 0.56, "danceability": 0.62, "acousticness": 0.71,
    }

    score, reasons = score_song(prefs, song)
    assert isinstance(score, float)
    assert len(reasons) > 0
