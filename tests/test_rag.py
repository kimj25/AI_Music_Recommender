"""
Tests for the RAG Engine (SongRAG).

Uses a small in-memory fixture — no real song data required.
"""

import json
import os
import tempfile
import pytest

MOCK_SONGS = [
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill",
        "energy": 0.42, "tempo_bpm": 78.0, "valence": 0.56,
        "danceability": 0.62, "acousticness": 0.71,
    },
    {
        "id": 12, "title": "Bass Drop Protocol", "artist": "Voltage Surge",
        "genre": "edm", "mood": "energetic",
        "energy": 0.95, "tempo_bpm": 140.0, "valence": 0.62,
        "danceability": 0.91, "acousticness": 0.03,
    },
    {
        "id": 13, "title": "Moonlit Sonata", "artist": "Clara Voss",
        "genre": "classical", "mood": "calm",
        "energy": 0.22, "tempo_bpm": 66.0, "valence": 0.78,
        "danceability": 0.30, "acousticness": 0.97,
    },
    {
        "id": 6, "title": "Spacewalk Thoughts", "artist": "Orbit Bloom",
        "genre": "ambient", "mood": "chill",
        "energy": 0.28, "tempo_bpm": 60.0, "valence": 0.65,
        "danceability": 0.41, "acousticness": 0.92,
    },
]

MOCK_DESCRIPTIONS = {
    "Midnight Coding": "A lo-fi hip-hop track with warm vinyl crackle perfect for late-night studying and focused work.",
    "Bass Drop Protocol": "A high-energy EDM banger with massive drops and pounding basslines made for dance floors and intense workouts.",
    "Moonlit Sonata": "A delicate classical solo piano piece for quiet evenings, sleep, and gentle reflection.",
    "Spacewalk Thoughts": "A dreamy ambient piece with floating soundscapes perfect for meditation and deep relaxation.",
}


@pytest.fixture
def rag(tmp_path):
    from src.rag import SongRAG
    desc_file = tmp_path / "descriptions.json"
    desc_file.write_text(json.dumps(MOCK_DESCRIPTIONS))
    return SongRAG(MOCK_SONGS, str(desc_file))


def test_search_returns_list(rag):
    results = rag.search("late night study music")
    assert isinstance(results, list)


def test_search_returns_song_dicts(rag):
    results = rag.search("workout energy", n=2)
    for song in results:
        assert "title" in song
        assert "genre" in song
        assert "energy" in song


def test_search_respects_n_limit(rag):
    results = rag.search("music", n=2)
    assert len(results) <= 2


def test_search_returns_at_least_one_result(rag):
    results = rag.search("something chill and relaxing")
    assert len(results) >= 1


def test_search_results_are_full_song_dicts(rag):
    results = rag.search("piano classical calm")
    required_keys = {"id", "title", "artist", "genre", "mood", "energy"}
    for song in results:
        assert required_keys.issubset(song.keys())


def test_search_with_different_queries_returns_results(rag):
    queries = [
        "something for late night coding",
        "high energy dance music",
        "peaceful meditation",
    ]
    for query in queries:
        results = rag.search(query, n=2)
        assert len(results) >= 1, f"No results for query: {query}"
