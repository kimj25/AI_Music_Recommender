from src.preference_model import PreferenceModel

model = PreferenceModel()

inputs = [
    "I want something upbeat and happy for a birthday party",
    "music for a late night drive through the city",
    "music",
]

print("\n=== Confidence Scoring Demo ===\n")

for text in inputs:
    result = model.extract(text)
    confidence = result["confidence"]
    label = "high" if confidence >= 0.85 else "low" if confidence < 0.7 else "medium"
    print(f'Input:      "{text}"')
    print(f'Genre:      {result["genre"]}  |  Mood: {result["mood"]}')
    print(f'Confidence: {confidence} ({label})')
    print()
