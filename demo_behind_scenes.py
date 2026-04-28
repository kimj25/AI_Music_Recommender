from src.preference_model import PreferenceModel
from src.rag import SongRAG
from src.recommender import load_songs, recommend_songs

songs = load_songs("data/songs.csv")
model = PreferenceModel()
rag = SongRAG(songs, "data/song_descriptions.json")

query = "music for a late night drive through the city"
print(f'\nInput: "{query}"\n')

print("--- Step 1: Specialized Model ---")
print("Converting your words into a structured profile...\n")
profile = model.extract(query)
print(f"Genre:      {profile['genre']}")
print(f"Mood:       {profile['mood']}")
print(f"Energy:     {profile['energy']}")
print(f"Valence:    {profile['valence']}")
print(f"Tempo:      {profile['tempo_bpm']} bpm")
print(f"Confidence: {profile['confidence']}")

print("\n--- Step 2: RAG Engine ---")
print("Input is vague — searching by meaning instead of exact labels...\n")
rag_results = rag.search(query, n=5)
for song in rag_results:
    print(f"  {song['title']} ({song['genre']})")

print("\n--- Step 3: Scoring Algorithm ---")
print("Ranking all songs against your profile...\n")
top_songs = recommend_songs(profile, songs, k=3)
for song, score, reasons in top_songs:
    print(f"  {song['title']} ({song['genre']}) — score: {score:.1f}/11.0")
    print(f"    {reasons}")
