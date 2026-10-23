"""
Phase 4 — Self-reflective verification layer (adapted from Self-RAG, Asai et al., 2023).

The generated answer is split into atomic claims. Each claim is checked against the
retrieved evidence with a Natural-Language-Inference cross-encoder: a claim is
"supported" if any evidence passage entails it above a threshold. Unsupported claims
are flagged and removed (or hedged), and the final answer is returned with an
explicit per-claim grounding report and a faithfulness score
= supported_claims / total_claims.

Backends:
  * NLIVerifier   — cross-encoder/nli-deberta-v3-base (default; ~180 MB)
  * LLMVerifier   — asks the generator LLM to judge support (fallback when no torch)
  * OverlapVerifier — lexical overlap stub for offline tests
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Chunk
from .generation import Answer

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    score: float
    best_evidence: int | None  # 1-based index into answer.evidence


@dataclass
class VerificationReport:
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        return sum(c.supported for c in self.claims) / len(self.claims) if self.claims else 0.0

    @property
    def n_unsupported(self) -> int:
        return sum(not c.supported for c in self.claims)

    def to_dict(self) -> dict:
        return {"faithfulness": round(self.faithfulness, 3), "n_claims": len(self.claims),
                "n_unsupported": self.n_unsupported,
                "claims": [c.__dict__ for c in self.claims]}


def split_claims(text: str, min_words: int = 4) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in _SENT_SPLIT.split(text)]
    sents = [s for s in sents if len(s.split()) >= min_words]
    # drop meta sentences that are not factual claims
    return [s for s in sents if not re.match(r"(?i)^(the retrieved evidence does not|i (cannot|can't)|note:)", s)]


class OverlapVerifier:
    name = "overlap"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def _score(self, claim: str, passage: str) -> float:
        cw = {w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 3}
        pw = set(re.findall(r"[a-z0-9]+", passage.lower()))
        return len(cw & pw) / max(len(cw), 1)

    def entail(self, claim: str, passage: str) -> float:
        return self._score(claim, passage)


class NLIVerifier:
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.3, device=None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        self.threshold = threshold
        # label order for this model family: contradiction, entailment, neutral
        self.entail_idx = 1

    def entail(self, claim: str, passage: str) -> float:
        import numpy as np
        logits = self.model.predict([(passage, claim)])[0]
        p = np.exp(logits - logits.max()); p = p / p.sum()
        return float(p[self.entail_idx])


class LLMVerifier:
    name = "llm"

    def __init__(self, llm, threshold: float = 0.5):
        self.llm, self.threshold = llm, threshold

    def entail(self, claim: str, passage: str) -> float:
        r = self.llm.chat("You are a strict fact checker. Reply with only YES or NO.",
                          f"Passage: {passage}\n\nClaim: {claim}\n\nIs the claim fully supported by the passage?",
                          max_new_tokens=3)
        return 1.0 if r.strip().upper().startswith("YES") else 0.0


class Verifier:
    """Wraps a backend and applies the self-reflective check to an Answer."""

    def __init__(self, backend, strip_citations: bool = True):
        self.b = backend
        self.strip = strip_citations

    def check(self, answer: Answer) -> VerificationReport:
        rep = VerificationReport()
        if not answer.evidence:
            return rep
        for claim in split_claims(answer.text):
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", claim)]
            plain = re.sub(r"\s*\[\d+\]", "", claim).strip() if self.strip else claim
            # check against cited passages first, then all evidence
            order = [i for i in cited if 1 <= i <= len(answer.evidence)] + \
                    [i for i in range(1, len(answer.evidence) + 1) if i not in cited]
            best_i, best_s = None, 0.0
            for i in order:
                s = self.b.entail(plain, answer.evidence[i - 1].text)
                if s > best_s:
                    best_i, best_s = i, s
                if s >= self.b.threshold:
                    break
            rep.claims.append(ClaimCheck(plain, best_s >= self.b.threshold, round(best_s, 3), best_i))
        return rep

    def revise(self, answer: Answer, rep: VerificationReport, policy: str = "remove") -> Answer:
        """Return a new Answer where unsupported claims are removed or hedged."""
        kept = []
        for c in rep.claims:
            if c.supported:
                kept.append(f"{c.claim} [{c.best_evidence}]" if c.best_evidence else c.claim)
            elif policy == "hedge":
                kept.append(f"(Unverified) {c.claim}")
        if kept:
            text = " ".join(kept)
        else:  # nothing verified: keep the draft but flag it, rather than silently abstaining
            text = "Low confidence (no claim could be verified against the evidence): " + answer.text
        new = Answer(answer.question, text, answer.evidence, answer.scores, answer.latency_s, mode=answer.mode + "+verify")
        new.verification = rep.to_dict()
        return new
