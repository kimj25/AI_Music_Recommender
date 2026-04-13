from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: float
    target_tempo_bpm: float

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv

    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against a user preference profile.
    Returns (score, reasons) where reasons explains each contribution.
    Max possible score: 11.0
    """
    score = 0.0
    reasons = []

    # Categorical matches
    if song["genre"] == user_prefs.get("genre"):
        score += 3.0
        reasons.append(f"genre match (+3.0)")
    if song["mood"] == user_prefs.get("mood"):
        score += 2.0
        reasons.append(f"mood match (+2.0)")
    if user_prefs.get("likes_acoustic") and song["acousticness"] > 0.6:
        score += 1.0
        reasons.append(f"acoustic preference (+1.0)")
    elif user_prefs.get("likes_acoustic") == False and song["acousticness"] > 0.6:
        score -= 1.0
        reasons.append(f"acoustic mismatch (-1.0)")

    # Continuous similarity
    energy_pts = 2.0 * (1 - abs(user_prefs.get("energy", 0) - song["energy"]))
    score += energy_pts
    reasons.append(f"energy similarity (+{energy_pts:.2f})")

    valence_pts = 1.5 * (1 - abs(user_prefs.get("valence", 0) - song["valence"]))
    score += valence_pts
    reasons.append(f"valence similarity (+{valence_pts:.2f})")

    tempo_pts = max(0.0, 1.0 * (1 - abs(user_prefs.get("tempo_bpm", 0) - song["tempo_bpm"]) / 100))
    score += tempo_pts
    reasons.append(f"tempo similarity (+{tempo_pts:.2f})")

    dance_pts = song["danceability"] * 0.5
    score += dance_pts
    reasons.append(f"danceability bonus (+{dance_pts:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = [
        (song, score, ", ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
