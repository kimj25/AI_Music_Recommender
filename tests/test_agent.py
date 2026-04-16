"""
Tests for the Agentic Orchestrator (MusicAgent).

All tests mock the Anthropic API and sub-components to avoid real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


def _make_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _make_tool_then_text_response(tool_name: str, tool_input: dict, tool_use_id: str, final_text: str):
    """Returns (tool_use_response, final_text_response) pair."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_use_id

    tool_response = MagicMock()
    tool_response.stop_reason = "tool_use"
    tool_response.content = [tool_block]

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = final_text
    text_response = MagicMock()
    text_response.stop_reason = "end_turn"
    text_response.content = [text_block]

    return tool_response, text_response


@pytest.fixture
def agent():
    """Create a MusicAgent with all external dependencies mocked."""
    with patch("src.agent.PreferenceModel"), \
         patch("src.agent.SongRAG"), \
         patch("src.agent.anthropic.Anthropic"), \
         patch("src.agent.load_songs", return_value=[]):
        from src.agent import MusicAgent
        a = MusicAgent("data/songs.csv", "data/song_descriptions.json")
        a.songs = []
        a.songs_by_title = {}
        return a


def test_chat_returns_string_response(agent):
    agent.client.messages.create.return_value = _make_text_response("Here are some songs!")

    response, _ = agent.chat("something chill", [])

    assert isinstance(response, str)
    assert len(response) > 0


def test_chat_returns_updated_history(agent):
    agent.client.messages.create.return_value = _make_text_response("Great picks coming up!")

    _, history = agent.chat("I want something upbeat", [])

    assert isinstance(history, list)
    assert len(history) == 2  # user message + assistant message


def test_chat_appends_to_existing_history(agent):
    prior = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [MagicMock(type="text", text="Hi!")]},
    ]
    agent.client.messages.create.return_value = _make_text_response("More songs!")

    _, history = agent.chat("something else", prior)

    assert len(history) > len(prior)


def test_chat_handles_tool_use_loop(agent):
    """Agent should loop through tool use and return final text."""
    tool_resp, text_resp = _make_tool_then_text_response(
        tool_name="get_recommendations",
        tool_input={"user_prefs": {"genre": "lofi", "mood": "chill", "energy": 0.4,
                                   "likes_acoustic": True, "valence": 0.55, "tempo_bpm": 78}},
        tool_use_id="tool_123",
        final_text="Here are your top picks!",
    )
    agent.client.messages.create.side_effect = [tool_resp, text_resp]
    agent.songs = []

    response, _ = agent.chat("lofi study music", [])

    assert response == "Here are your top picks!"
    assert agent.client.messages.create.call_count == 2


def test_handle_tool_get_recommendations(agent):
    """_handle_tool should call recommend_songs and return valid JSON."""
    from src.recommender import load_songs
    agent.songs = load_songs("data/songs.csv")

    prefs = {
        "genre": "lofi", "mood": "chill", "energy": 0.42,
        "likes_acoustic": True, "valence": 0.56, "tempo_bpm": 78,
    }
    result = json.loads(agent._handle_tool("get_recommendations", {"user_prefs": prefs, "k": 3}))

    assert len(result) == 3
    assert all("song" in r and "score" in r and "reasons" in r for r in result)


def test_handle_tool_get_song_details_found(agent):
    agent.songs_by_title = {
        "Midnight Coding": {"id": 2, "title": "Midnight Coding", "genre": "lofi"}
    }
    result = json.loads(agent._handle_tool("get_song_details", {"title": "Midnight Coding"}))

    assert result["title"] == "Midnight Coding"
    assert result["genre"] == "lofi"


def test_handle_tool_get_song_details_not_found(agent):
    agent.songs_by_title = {}
    result = json.loads(agent._handle_tool("get_song_details", {"title": "Nonexistent Song"}))

    assert "error" in result


def test_handle_tool_unknown_tool(agent):
    result = json.loads(agent._handle_tool("unknown_tool", {}))

    assert "error" in result
