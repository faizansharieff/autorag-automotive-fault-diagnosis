"""
Phase 5 — Evaluation protocol (Table 7.2 of the proposal).

Test set
  Built from held-out Q&A pairs (gold answers) plus recall records turned into
  questions ("What is the defect and remedy for <vehicle> <component>?") whose gold
  answer is the recall text. Q&A gold answers are REMOVED from the retrieval corpus so
  the system cannot simply copy them (generalisation test); recall records stay in the
  corpus (look-up / grounding test, measured by retrieval_hit).

Metrics (per answer)
  accuracy      — LLM-judge correctness vs gold (0/0.5/1) + ROUGE-L / BERTScore-free token F1
  relevance     — LLM-judge answer-to-question relevance (1-5 -> 0-1)
  faithfulness  — fraction of claims entailed by retrieved evidence (NLI, independent of the
                  verifier used inside S3); for llm_only there is no evidence -> N/A
  citation_rate — fraction of claims carrying a valid citation
  retrieval hit — gold-source chunk appears in top-k (RAG systems only)
  latency_s     — wall-clock seconds
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd

from .corpus import Chunk
from .generation import Answer
from .verify import Verifier, split_claims

JUDGE_SYS = ("You are an expert automotive service engineer grading diagnostic answers. "
             "Reply with JSON only: {\"correct\": 0|0.5|1, \"relevance\": 1-5, \"reason\": \"...\"}. "
             "correct=1 if the answer gives the same cause/remedy as the reference, 0.5 if partially, 0 if wrong "
             "or unsupported. relevance = how well the answer addresses the question.")


def build_testset(chunks: list[Chunk], n_qa: int = 60, n_recall: int = 40, seed: int = 7) -> tuple[list[dict], list[Chunk]]:
    """Returns (test_items, remaining_corpus). Each item: {question, gold, gold_chunk_id, source}."""
    rng = random.Random(seed)
    qa = [c for c in chunks if c.source == "qa" and c.text.startswith("Question:")]
    rc = [c for c in chunks if c.source == "recall"]
    rng.shuffle(qa); rng.shuffle(rc)
    items, held = [], set()
    for c in qa[:n_qa]:
        m = re.match(r"Question:\s*(.*?)\s*Answer:\s*(.*)", c.text, flags=re.S)
        if not m:
            continue
        items.append({"question": m.group(1).strip(), "gold": m.group(2).strip(),
                      "gold_chunk_id": c.chunk_id, "source": "qa"})
        held.add(c.doc_id)
    for c in rc[:n_recall]:
        m = c.meta
        veh = f"{m.get('year')} {str(m.get('make')).title()} {str(m.get('model')).title()}"
        comp = m.get("component", "component")
        items.append({"question": f"What is the known defect, its consequence and the remedy for the {comp} on a {veh}?",
                      "gold": c.text, "gold_chunk_id": c.chunk_id, "source": "recall"})
        # recall items are look-up questions: the record stays in the corpus so that
        # retrieval_hit measures whether the system actually finds and cites it.
    remaining = [c for c in chunks if c.doc_id not in held]
    return items, remaining


# ---------------------------------------------------------------------------
def token_f1(pred: str, gold: str) -> float:
    p = re.findall(r"[a-z0-9]+", pred.lower()); g = re.findall(r"[a-z0-9]+", gold.lower())
    if not p or not g:
        return 0.0
    from collections import Counter
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def rouge_l(pred: str, gold: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer.RougeScorer(["rougeL"]).score(gold, pred)["rougeL"].fmeasure
    except Exception:
        return token_f1(pred, gold)


def llm_judge(llm, question: str, gold: str, pred: str) -> dict:
    user = f"Question: {question}\n\nReference answer: {gold}\n\nCandidate answer: {pred}"
    try:
        raw = llm.chat(JUDGE_SYS, user, max_new_tokens=60)
        js = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group())
        return {"correct": float(js.get("correct", 0)), "relevance": (float(js.get("relevance", 1)) - 1) / 4}
    except Exception:
        return {"correct": token_f1(pred, gold) > 0.3 and 1.0 or 0.0, "relevance": min(1.0, token_f1(pred, question) * 3)}


def citation_rate(a: Answer) -> float | None:
    if not a.evidence:
        return None
    claims = split_claims(a.text)
    if not claims:
        return 0.0
    n = len(a.evidence)
    ok = sum(any(1 <= int(x) <= n for x in re.findall(r"\[(\d+)\]", c)) for c in claims)
    return ok / len(claims)


def evaluate(rag, items: list[dict], judge_llm, faith_verifier: Verifier, systems=None,
             out_dir: str | Path = "results", limit: int | None = None) -> pd.DataFrame:
    systems = systems or rag.SYSTEMS
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "per_question.csv"
    rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
    done = {(r.get("question"), r.get("system")) for r in rows}
    if rows: print(f"resuming: {len(rows)} rows already done")
    for n_it, it in enumerate(items[:limit]):
        for s in systems:
            if (it["question"], s) in done:
                continue
            try:
                a = rag.run(s, it["question"])
            except Exception as e:  # keep the run going
                rows.append({"system": s, "question": it["question"], "error": str(e)}); continue
            j = llm_judge(judge_llm, it["question"], it["gold"], a.text)
            faith = faith_verifier.check(a).faithfulness if a.evidence else None
            hit = (it["gold_chunk_id"].split("#")[0] in {c.doc_id for c in a.evidence}) if a.evidence else None
            rows.append({
                "system": s, "src": it["source"], "question": it["question"],
                "answer": a.text, "gold": it["gold"],
                "correct": j["correct"], "relevance": j["relevance"],
                "rougeL": rouge_l(a.text, it["gold"]),
                "faithfulness": faith, "citation_rate": citation_rate(a),
                "retrieval_hit": hit, "latency_s": round(a.latency_s, 3),
                "n_evidence": len(a.evidence),
                "n_unsupported": (a.verification or {}).get("n_unsupported"),
            })
        pd.DataFrame(rows).to_csv(ckpt, index=False)  # checkpoint after every question
        print(f"[{n_it+1}/{len(items[:limit])}] done", flush=True)
    df = pd.DataFrame(rows)
    for col in ("retrieval_hit", "faithfulness", "citation_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = (df.groupby("system")[["correct", "relevance", "rougeL", "faithfulness", "citation_rate",
                                     "retrieval_hit", "latency_s"]].mean(numeric_only=True)
               .reindex(list(systems)).round(3))
    summary["hallucination_rate"] = (1 - summary["faithfulness"]).round(3)
    summary.to_csv(out_dir / "summary.csv")
    return df, summary


def plot_summary(summary: pd.DataFrame, out_dir: str | Path = "results") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    cols = [c for c in ["correct", "relevance", "faithfulness", "citation_rate", "retrieval_hit"] if c in summary]
    ax = summary[cols].plot(kind="bar", figsize=(9, 4.5), rot=15, width=0.8)
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.set_title("System comparison (higher is better)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "fig_system_comparison.png", dpi=200); plt.close()
    ax = summary[["latency_s"]].plot(kind="bar", figsize=(6, 3.5), rot=15, legend=False, color="#888")
    ax.set_ylabel("seconds"); ax.set_title("Mean latency per query")
    plt.tight_layout(); plt.savefig(out_dir / "fig_latency.png", dpi=200); plt.close()
