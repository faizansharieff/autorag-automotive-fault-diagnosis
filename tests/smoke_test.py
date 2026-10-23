"""Offline end-to-end check: no network, no torch. Run: python -I tests/smoke_test.py"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autorag.corpus import build_corpus, corpus_stats
from autorag.retrieval import Retriever
from autorag.generation import StubChat, Generator
from autorag.verify import Verifier, OverlapVerifier
from autorag.pipeline import AutoRAG
from autorag.evaluate import build_testset, evaluate, plot_summary

random.seed(1)
makes = [("honda", "civic", 2016), ("toyota", "camry", 2017), ("ford", "f-150", 2018)]
comps = ["engine", "transmission", "brakes", "electrical"]
raw = []
for i in range(120):
    mk, md, yr = random.choice(makes); cp = random.choice(comps)
    raw.append({"id": f"c{i}", "source": "complaint",
                "text": f"Vehicle: {yr} {mk.title()} {md.title()}. Component: {cp}. Complaint: the {cp} started failing at highway speed and the warning light came on. The dealer could not reproduce the fault number {i}.",
                "meta": {"make": mk, "model": md, "year": yr, "component": cp, "severity": "normal"}})
for i in range(40):
    mk, md, yr = random.choice(makes); cp = random.choice(comps)
    raw.append({"id": f"r{i}", "source": "recall",
                "text": f"Vehicle: {yr} {mk.title()} {md.title()}. Recall 21V{i:03d}. Component: {cp}. Defect: the {cp} control module may lose power. Consequence: loss of {cp} function increases crash risk. Remedy: dealers will replace the {cp} module free of charge.",
                "meta": {"make": mk, "model": md, "year": yr, "component": cp, "severity": "high"}})
for i in range(80):
    cp = random.choice(comps)
    raw.append({"id": f"q{i}", "source": "qa",
                "text": f"Question: Why does my {cp} make a grinding noise when cold number {i}?\nAnswer: A grinding noise from the {cp} when cold usually indicates worn bearings or low fluid. Check the fluid level and inspect for metal shavings, then replace the worn component.",
                "meta": {"make": None, "model": None, "year": None, "component": cp, "severity": "normal"}})
raw.append({"id": "m0", "source": "manual", "meta": {"make": "honda", "file": "civic_service.txt"},
            "text": " ".join(f"Section {k}: To inspect the brakes remove the wheel, measure pad thickness; replace pads below 3 mm. Torque the caliper bolts to 35 Nm." for k in range(30))})

chunks = build_corpus(raw)
print("corpus:", corpus_stats(chunks))
items, corpus = build_testset(chunks, n_qa=5, n_recall=5)
print("test items:", len(items), "| corpus after hold-out:", len(corpus))

ret = Retriever(corpus, backend="tfidf")
gen = Generator(StubChat())
ver = Verifier(OverlapVerifier())
rag = AutoRAG(ret, gen, ver, k=4)

print("\n--- demo answer ---")
print(rag.answer("2016 Honda Civic brakes grinding when cold, what should I check?"))

df, summary = evaluate(rag, items, StubChat(), ver, out_dir="/tmp/autorag_smoke")
print("\n", summary)
plot_summary(summary, "/tmp/autorag_smoke")
print("\nOK — files in /tmp/autorag_smoke")
