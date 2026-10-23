"""
End-to-end systems under comparison (Phase 5 baselines):

  S0  llm_only          — no retrieval
  S1  vanilla_rag       — single source (complaints only), no metadata, no verification
  S2  multisource_rag   — four sources, metadata-aware retrieval, source attribution
  S3  multisource_rag+V — S2 + self-reflective verification layer
"""
from __future__ import annotations

import time

from .corpus import Chunk
from .generation import Answer, Generator
from .retrieval import Retriever
from .verify import Verifier


class AutoRAG:
    def __init__(self, retriever: Retriever, generator: Generator, verifier: Verifier | None = None, k: int = 5):
        self.r, self.g, self.v, self.k = retriever, generator, verifier, k

    def llm_only(self, q: str) -> Answer:
        return self.g.answer_plain(q)

    def vanilla_rag(self, q: str, source: str = "complaint") -> Answer:
        t0 = time.time()
        ev = self.r.search(q, k=self.k, sources=[source], metadata_aware=False)
        a = self.g.answer_rag(q, ev, mode="vanilla_rag")
        a.latency_s = time.time() - t0
        return a

    def multisource_rag(self, q: str) -> Answer:
        t0 = time.time()
        ev = self.r.search(q, k=self.k, sources=None, metadata_aware=True)
        a = self.g.answer_rag(q, ev, mode="multisource_rag")
        a.latency_s = time.time() - t0
        return a

    def multisource_rag_verified(self, q: str, policy: str = "remove") -> Answer:
        assert self.v is not None, "verifier not set"
        t0 = time.time()
        a = self.multisource_rag(q)
        rep = self.v.check(a)
        out = self.v.revise(a, rep, policy=policy)
        out.latency_s = time.time() - t0
        return out

    SYSTEMS = ("llm_only", "vanilla_rag", "multisource_rag", "multisource_rag_verified")

    def run(self, system: str, q: str) -> Answer:
        return getattr(self, system)(q)

    def answer(self, q: str) -> str:
        """Pretty, user-facing answer (final system)."""
        a = self.multisource_rag_verified(q) if self.v else self.multisource_rag(q)
        rep = a.verification or {}
        lines = [a.text, "", "Sources:", a.sources_block()]
        if rep:
            lines += ["", f"Grounding: {rep['n_claims'] - rep['n_unsupported']}/{rep['n_claims']} claims verified "
                          f"(faithfulness {rep['faithfulness']:.2f})"]
        return "\n".join(lines)
