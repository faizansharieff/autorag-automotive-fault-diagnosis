"""
Phase 3a — Retrieval.

Dense retrieval with Sentence-Transformers embeddings indexed in FAISS
(Karpukhin et al., 2020). Retrieval is metadata-aware: results can be
filtered by source, make/model or component, and the query is parsed for
vehicle mentions so the right manufacturer's records are preferred.

A TF-IDF backend is included so the pipeline runs on machines without
torch (used for offline tests and as a sparse baseline).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .data import COMPONENT_KEYWORDS, MAKES


def parse_query_meta(query: str) -> dict:
    """Extract make / year / component hints from the user's question."""
    q = query.lower()
    make = next((m for m in MAKES if re.search(rf"\b{re.escape(m)}\b", q)), None)
    year = re.search(r"\b(19[89]\d|20[0-3]\d)\b", q)
    comp, best = None, 0
    for c, kws in COMPONENT_KEYWORDS.items():
        s = sum(q.count(k) for k in kws)
        if s > best:
            comp, best = c, s
    return {"make": make, "year": int(year.group()) if year else None, "component": comp}


class Retriever:
    def __init__(self, chunks: list[Chunk], backend: str = "dense",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.chunks = chunks
        self.backend = backend
        self.model_name = model_name
        self._texts = [c.text for c in chunks]
        if backend == "dense":
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer(model_name, device=device)
            emb = self.model.encode(self._texts, batch_size=64, show_progress_bar=True,
                                    normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            self.emb = emb
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
            self.mat = self.vec.fit_transform(self._texts)
        else:
            raise ValueError(backend)

    # -- scoring ------------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        if self.backend == "dense":
            q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            scores, idx = self.index.search(q, len(self.chunks))
            out = np.zeros(len(self.chunks), dtype="float32")
            out[idx[0]] = scores[0]
            return out
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.vec.transform([query]), self.mat)[0]

    # -- public -------------------------------------------------------------
    def search(self, query: str, k: int = 5, sources: list[str] | None = None,
               metadata_aware: bool = True, meta_boost: float = 0.08) -> list[tuple[Chunk, float]]:
        scores = self._scores(query).copy()
        if sources:
            allowed = set(sources)
            for i, c in enumerate(self.chunks):
                if c.source not in allowed:
                    scores[i] = -1e9
        if metadata_aware:
            hint = parse_query_meta(query)
            for i, c in enumerate(self.chunks):
                m = c.meta or {}
                if hint["make"] and m.get("make") == hint["make"]:
                    scores[i] += meta_boost
                if hint["component"] and m.get("component") == hint["component"]:
                    scores[i] += meta_boost / 2
                if hint["year"] and m.get("year") == hint["year"]:
                    scores[i] += meta_boost / 2
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > -1e8]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model_name": self.model_name,
                         "emb": getattr(self, "emb", None)}, f)
