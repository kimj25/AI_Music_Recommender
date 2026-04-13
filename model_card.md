# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  : **MusicMate Finder 1.0**

---

## 2. Intended Use  

MusicMate Finder 1.0 is a rule-based song recommender designed to recommend musics using a scored preference profile. Given a user's stated preferences — genre, mood, energy level, acoustic taste, valence, and tempo — it ranks a catalog of songs and returns the top matches.
---

## 3. How the Model Works  

Every song gets a score based on how well it matches the user's preferences. Genre and mood matches add the most points (+3 and +2), followed by how close the song's energy, positivity, and tempo are to what the user likes.The top five highest-scoring songs are returned with its score.
---

## 4. Data  

The catalog contains 17 songs across 14 genres — including pop, lofi, rock, jazz, classical, hip hop, EDM, folk, and ambient — and 13 distinct mood labels. Genres like country, R&B, Latin, and K-pop are absent currently, and will be incorporated in the future updates.

---

## 5. Strengths  

The system works best for users with a clear, strong preference profile — for example, someone who knows they want chill lofi at low energy. The per-song score explanations also make it easy to see exactly why each song was recommended.

---

## 6. Limitations and Bias 

One of the limitations and bias:
Genre match (+3.0) and mood match (+2.0) together account for 45% of the maximum possible score, and both use exact string matching. This means a song that is sonically close to a user's taste but belongs to a slightly different genre or mood will consistently rank lower than a weaker overall match. Combined with a small catalog where most genres have only one song, the scoring creates a filter bubble that repeatedly surfaces the same narrow slice of music rather than encouraging discovery.

---

## 7. Evaluation  

Three user profiles were tested — a calm lofi listener, a high-energy dance listener, and a deep rock listener — each with distinct genre, mood, energy, and tempo preferences. For each profile, we checked whether the top-ranked songs matched the expected genre and mood and whether the score explanations correctly reflected what drove each result.

The most surprising finding was that songs with no genre or mood match could still rank highly due to strong energy and valence alignment, which revealed how much the continuous features contribute when categorical matches are absent. Edge cases were also tested, including profiles with missing preference fields, to confirm the system returned results without errors.

---

## 8. Future Work  

- Add mismatch penalties so songs that actively conflict with user preferences score lower, not just zero
- Expand the catalog to include underrepresented genres like R&B, country, and Latin
- Add a diversity constraint to prevent the same genre from filling multiple top-5 slots
- Allow users to adjust feature weights so they can control how much genre vs. energy matters to them

---

## 9. Personal Reflection  

Building this project showed how powerful AI tools can be for accelerating development — from generating scoring logic to drafting test cases quickly. However, I learned that AI doesn't replace careful review: each test case still needed to be checked manually to make sure it was actually testing the right behavior and not just passing for the wrong reasons. The biggest takeaway is that double-checking is essential when working with AI-generated code, because the output can look correct on the surface while missing subtle edge cases underneath. 
