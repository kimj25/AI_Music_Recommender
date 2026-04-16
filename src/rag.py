"""
RAG Engine: Semantic Song Search

Indexes song descriptions as vector embeddings in ChromaDB and enables
natural language queries like "late night study vibes" or "music for a rainy
afternoon" — queries that exact genre/mood matching can't handle.

Acts as a high-recall pre-filter; the scoring algorithm provides precision ranking.
"""

import json
import chromadb
from chromadb.utils import embedding_functions


class SongRAG:
    """
    Semantic search engine over the song catalog.

    Stores rich natural language descriptions per song as embeddings in ChromaDB.
    At query time, embeds the user query and returns songs whose descriptions
    are most semantically similar.

    Embedding model: sentence-transformers/all-MiniLM-L6-v2 (ChromaDB default).
    Vector store: ChromaDB in-memory (no external server required).
    """

    COLLECTION_NAME = "songs"

    def __init__(self, songs: list[dict], descriptions_path: str) -> None:
        """
        Args:
            songs: List of song dicts loaded from songs.csv
            descriptions_path: Path to song_descriptions.json
        """
        self.songs_by_title = {s["title"]: s for s in songs}
        self.descriptions = self._load_descriptions(descriptions_path)

        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
        )
        self._index()

    def _load_descriptions(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _index(self) -> None:
        if self._collection.count() > 0:
            return  # Already indexed in this session

        documents, ids, metadatas = [], [], []
        for title, description in self.descriptions.items():
            song = self.songs_by_title.get(title)
            if not song:
                continue
            documents.append(description)
            ids.append(str(song["id"]))
            metadatas.append({
                "title": title,
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
            })

        if documents:
            self._collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def search(self, query: str, n: int = 8) -> list[dict]:
        """
        Semantic search over song descriptions.

        Args:
            query: Natural language query, e.g. "something for a late-night drive"
            n: Maximum number of results to return

        Returns:
            List of song dicts ordered by semantic similarity (most similar first)
        """
        n = min(n, self._collection.count())
        if n == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=n)

        matched = []
        for meta in results["metadatas"][0]:
            song = self.songs_by_title.get(meta["title"])
            if song:
                matched.append(song)
        return matched
