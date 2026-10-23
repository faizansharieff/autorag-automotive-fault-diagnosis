"""
Phase 2 — Preprocessing and corpus construction.

Cleans raw documents, de-duplicates, splits long texts into retrieval-sized
chunks and attaches metadata (source, make, model, year, component, severity)
so retrieval can filter and every answer can cite its origin.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str          # complaint | qa | recall | manual
    text: str
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        m = self.meta
        veh = " ".join(str(x) for x in (m.get("year"), m.get("make"), m.get("model")) if x)
        tag = {"complaint": "NHTSA complaint", "recall": "NHTSA recall",
               "qa": "Diagnostic Q&A", "manual": f"Manual {m.get('file', '')}"}.get(self.source, self.source)
        return f"[{self.chunk_id} | {tag}{' | ' + veh if veh else ''}]"


def clean_text(t: str) -> str:
    t = _CTRL.sub(" ", t)
    t = t.replace(" ", " ")
    t = re.sub(r"(?i)\*?\s*tl\*?\s*the\s*contact", "The contact", t)  # common NHTSA artefact
    t = _WS.sub(" ", t).strip()
    return t


def chunk_text(text: str, max_words: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
    return out


def build_corpus(raw_docs: list[dict], max_words: int = 220, overlap: int = 40,
                 min_words: int = 15) -> list[Chunk]:
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for d in raw_docs:
        text = clean_text(d["text"])
        key = text[:300].lower()
        if key in seen or len(text.split()) < min_words:
            continue
        seen.add(key)
        for j, piece in enumerate(chunk_text(text, max_words, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{d['id']}#{j}",
                doc_id=d["id"],
                source=d["source"],
                text=piece,
                meta={k: v for k, v in (d.get("meta") or {}).items()},
            ))
    return chunks


def save_corpus(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_corpus(path: str | Path) -> list[Chunk]:
    with open(path) as f:
        return [Chunk(**json.loads(line)) for line in f]


def corpus_stats(chunks: list[Chunk]) -> dict:
    import collections
    by_src = collections.Counter(c.source for c in chunks)
    words = [len(c.text.split()) for c in chunks]
    return {"chunks": len(chunks), "by_source": dict(by_src),
            "avg_words": round(sum(words) / max(len(words), 1), 1),
            "docs": len({c.doc_id for c in chunks})}
