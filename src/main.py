"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from tabulate import tabulate

try:
    from .recommender import load_songs, recommend_songs
except ImportError:
    from recommender import load_songs, recommend_songs

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    profiles = {
        "user_prefs": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.38,
            "likes_acoustic": True,
            "valence": 0.55,
            "tempo_bpm": 78,
        },
        # Taste profile: late-night focus listener
        # Prefers lo-fi or ambient tracks, calm/chill mood, low energy, acoustic leaning
        "Calm Ambient Lofi": {
            "genre": "lofi",
            "mood": "calm",
            "energy": 0.30,
            "likes_acoustic": True,
            "valence": 0.55,
            "tempo_bpm": 72,
        },
        # Taste profile: high-energy dance floor listener
        # Prefers EDM, energetic mood, high energy, high valence, fast tempo
        "High-energy Dance": {
            "genre": "edm",
            "mood": "energetic",
            "energy": 0.92,
            "likes_acoustic": False,
            "valence": 0.85,
            "tempo_bpm": 138,
        },
        # Taste profile: deep, intense rock listener
        # Prefers rock, intense mood, high energy, low valence, fast tempo
        "Deep Intense Rock": {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.90,
            "likes_acoustic": False,
            "valence": 0.35,
            "tempo_bpm": 150,
        },
    }

    for profile_name, user_prefs in profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\n Profile: {profile_name}")
        print("=" * 90)

        rows = []
        for i, (song, score, explanation) in enumerate(recommendations, start=1):
            reasons = "\n".join(f"  {r.strip()}" for r in explanation.split(","))
            rows.append([
                f"#{i}",
                f"{song['title']}\n{song['artist']}",
                f"{song['genre']}\n{song['mood']}",
                f"{score:.2f} / 11.0\n{score / 11.0:.0%} match",
                reasons,
            ])

        print(tabulate(
            rows,
            headers=["Rank", "Title / Artist", "Genre / Mood", "Score", "Score Reasons"],
            tablefmt="fancy_grid",
            colalign=("center", "left", "center", "center", "left"),
        ))
        print()


if __name__ == "__main__":
    main()
