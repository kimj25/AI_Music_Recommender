"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Taste profile: late-night focus listener
    # Prefers lo-fi or ambient tracks, calm/chill mood, low energy, acoustic leaning
    user_prefs = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
        "valence": 0.55,
        "tempo_bpm": 78,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    print("-" * 40)
    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"#{i}  {song['title']} by {song['artist']}")
        print(f"    Score : {score:.2f} / 11.0  ({score / 11.0:.0%} match)")
        print(f"    Why   : {explanation}")
        print("-" * 40)


if __name__ == "__main__":
    main()
