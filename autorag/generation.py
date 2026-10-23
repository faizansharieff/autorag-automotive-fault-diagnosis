"""
Phase 3b — Grounded generation with source attribution.

Three interchangeable LLM backends:
  * HFChat     — open model via transformers (default Qwen2.5-3B/7B-Instruct, 4-bit on Colab T4)
  * APIChat    — OpenAI-compatible or Anthropic API (set key via env var)
  * StubChat   — deterministic extractive stub for offline testing

The RAG prompt forces the model to answer ONLY from the numbered evidence
passages and to cite them as [1], [2] ... (evidence-aggregation in the spirit
of Fusion-in-Decoder, Izacard & Grave, 2021).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .corpus import Chunk

SYSTEM_RAG = (
    "You are an automotive diagnostic assistant for service engineers. Answer the question "
    "using ONLY the numbered evidence passages. After every factual statement cite the passage "
    "number in square brackets, e.g. [2]. If the evidence does not contain the answer, say "
    "'The retrieved evidence does not cover this' instead of guessing. Be concise: give the likely "
    "cause(s), how to confirm, and the recommended fix in 3-5 sentences of your own words; do not copy "
    "the passages verbatim or repeat 'Question:'/'Answer:' labels. Never invent part numbers or procedures."
)
SYSTEM_PLAIN = (
    "You are an automotive diagnostic assistant for service engineers. Answer concisely: likely "
    "cause(s), how to confirm, recommended fix."
)


@dataclass
class Answer:
    question: str
    text: str
    evidence: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_s: float = 0.0
    mode: str = "rag"
    verification: dict | None = None

    def cited_ids(self) -> list[int]:
        return sorted({int(n) for n in re.findall(r"\[(\d+)\]", self.text)})

    def sources_block(self) -> str:
        lines = []
        for i, c in enumerate(self.evidence, 1):
            lines.append(f"[{i}] {c.citation()}")
        return "\n".join(lines)


def build_rag_prompt(question: str, evidence: list[Chunk]) -> str:
    ev = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(evidence, 1))
    return f"Evidence passages:\n{ev}\n\nQuestion: {question}\nAnswer (cite passages):"


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class StubChat:
    """Offline stand-in: returns the top evidence sentences with citations."""
    name = "stub"

    def chat(self, system: str, user: str, max_new_tokens: int = 300) -> str:
        if "Evidence passages" not in user:
            return "Likely cause: a common fault in this component. Confirm with a scan tool and inspect the part. Fix: replace the faulty part."
        passages = re.findall(r"\[(\d+)\] \(\w+\) (.*?)(?=\n\n\[\d+\]|\n\nQuestion:)", user, flags=re.S)
        out = []
        for n, p in passages[:3]:
            sents = re.split(r"(?<=[.!?])\s", p.strip())
            body = " ".join(sents[-3:])[:300]  # tail sentences usually hold cause/remedy
            out.append(f"{body} [{n}]")
        return " ".join(out) or "The retrieved evidence does not cover this."


class HFChat:
    name = "hf"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = True,
                 device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = {"device_map": device_map, "dtype": torch.float16}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval()

    def chat(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class APIChat:
    """OpenAI-compatible (OpenAI, Groq, Together…) or Anthropic."""
    name = "api"

    def __init__(self, provider: str = "openai", model: str | None = None, base_url: str | None = None):
        self.provider = provider
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = model or "claude-haiku-4-5"
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"),
                                 base_url=base_url)
            self.model = model or "gpt-4o-mini"

    def chat(self, system: str, user: str, max_new_tokens: int = 350) -> str:
        if self.provider == "anthropic":
            r = self.client.messages.create(model=self.model, max_tokens=max_new_tokens, temperature=0,
                                            system=system, messages=[{"role": "user", "content": user}])
            return r.content[0].text.strip()
        r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=max_new_tokens,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        return r.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
class Generator:
    def __init__(self, llm):
        self.llm = llm

    def answer_plain(self, question: str) -> Answer:
        t0 = time.time()
        text = self.llm.chat(SYSTEM_PLAIN, question)
        return Answer(question, text, [], [], time.time() - t0, mode="llm_only")

    def answer_rag(self, question: str, evidence: list[tuple[Chunk, float]], mode: str = "rag") -> Answer:
        t0 = time.time()
        chunks = [c for c, _ in evidence]
        text = self.llm.chat(SYSTEM_RAG, build_rag_prompt(question, chunks))
        return Answer(question, text, chunks, [s for _, s in evidence], time.time() - t0, mode=mode)
