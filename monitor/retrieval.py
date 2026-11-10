"""Retrieval: chunk articles, turn chunks into vectors, and find related earlier coverage.

Offline stand-in for embeddings: TF-IDF vectors + cosine similarity in plain Python.
In production you would call an embedding model and keep the vectors in a vector store
(pgvector, Chroma, Snowflake or Databricks vector search). The interface stays the same:
add chunks with metadata, then search(query, filters) -> top-k chunks.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .data import find_counterparties

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "said", "that", "the", "this", "to", "was", "were", "will", "with",
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def chunk_text(text: str, max_words: int = 60, overlap_sentences: int = 1) -> list[str]:
    """Group whole sentences into chunks of about `max_words`, repeating the last
    sentence(s) of each chunk at the start of the next so context is not cut mid-thought."""
    chunks, current = [], []
    for sentence in split_sentences(text):
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    title: str
    published_at: str
    text: str
    counterparties: list[str] = field(default_factory=list)  # metadata used for filtering


class VectorIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._doc_freq: Counter = Counter()
        self._vectors: list[dict[str, float]] | None = None

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self._doc_freq.update(set(tokenize(chunk.text)))
        self._vectors = None  # IDF changed, so cached vectors are stale

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        return math.log((1 + n) / (1 + self._doc_freq[term])) + 1

    def embed(self, text: str) -> dict[str, float]:
        """TF-IDF vector, normalised to length 1 so a dot product = cosine similarity."""
        counts = Counter(tokenize(text))
        vector = {term: count * self._idf(term) for term, count in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, k: int = 2, counterparty: str | None = None,
               before: str | None = None, exclude_article_id: str | None = None,
               min_score: float = 0.05) -> list[tuple[float, Chunk]]:
        if self._vectors is None:
            self._vectors = [self.embed(c.text) for c in self.chunks]
        query_vector = self.embed(query)

        results = []
        for chunk, vector in zip(self.chunks, self._vectors):
            # Metadata filters run before scoring, like a WHERE clause in a vector DB.
            if counterparty and counterparty not in chunk.counterparties:
                continue
            if before and chunk.published_at >= before:  # never "retrieve the future"
                continue
            if exclude_article_id and chunk.article_id == exclude_article_id:
                continue
            score = sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items())
            if score >= min_score:
                results.append((round(score, 3), chunk))

        results.sort(key=lambda r: r[0], reverse=True)
        return results[:k]


def build_index(articles: list[dict], counterparties: list[dict], max_words: int = 60) -> VectorIndex:
    index = VectorIndex()
    for article in articles:
        full_text = f"{article['title']}. {article['body']}"
        # Tag every chunk with the counterparties named anywhere in its article (metadata enrichment).
        names = find_counterparties(full_text, counterparties)
        for i, text in enumerate(chunk_text(full_text, max_words=max_words)):
            index.add(Chunk(
                chunk_id=f"{article['article_id']}-{i}",
                article_id=article["article_id"],
                title=article["title"],
                published_at=article["published_at"],
                text=text,
                counterparties=names,
            ))
    return index
