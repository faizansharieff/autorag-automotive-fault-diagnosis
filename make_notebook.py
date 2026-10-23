import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Multi-Source RAG for Automotive Fault Diagnosis — Experiment Notebook
**Faizan Sharieff — Master's Thesis (upGrad)**

Runs the full five-phase pipeline on Google Colab (Runtime → Change runtime type → **T4 GPU**).

| Phase | Step | Cell |
|---|---|---|
| 1 | Collect 4 sources (NHTSA complaints, NHTSA recalls, diagnostic Q&A, manuals) | 2 |
| 2 | Clean, chunk, add metadata | 3 |
| 3 | Dense retrieval (MiniLM + FAISS) + grounded generation (Qwen2.5-3B-Instruct) | 4–5 |
| 4 | Self-reflective verification (NLI cross-encoder) | 6 |
| 5 | Evaluate 4 systems: LLM-only · vanilla RAG · multi-source RAG · + verification | 7–8 |

Total run time on a T4: ~45 min for 40 test questions. Results are checkpointed to Google Drive after every question, so a disconnected run resumes.""")

code("""#@title 1. Setup (≈3 min)
!pip -q install sentence-transformers faiss-cpu transformers accelerate bitsandbytes datasets rouge-score pypdf requests pandas matplotlib scipy
import os, sys
if not os.path.exists("autorag"):
    !git clone -q https://github.com/faizansharieff/autorag-automotive-fault-diagnosis.git repo && cp -r repo/autorag .
sys.path.insert(0, ".")
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/AutoRAG_results'; os.makedirs(OUT, exist_ok=True)
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")""")

code("""#@title 2. Phase 1 — collect data (≈5 min, cached after first run)
from autorag.data import fetch_nhtsa_complaints, fetch_nhtsa_recalls, load_diagnostic_qa, load_manuals, load_dtc_reference_manual
complaints = fetch_nhtsa_complaints()
recalls    = fetch_nhtsa_recalls()
# Diagnostic Q&A: EITHER upload the Kaggle CSV to data/qa.csv OR leave None to pull a public HF dataset
QA_PATH = "data/qa.csv" if os.path.exists("data/qa.csv") else None
qa = load_diagnostic_qa(QA_PATH)
# Manuals: drop public service-manual PDFs / .txt into data/manuals/
os.makedirs("data/manuals", exist_ok=True)
manuals = load_manuals("data/manuals") or load_dtc_reference_manual()
raw = complaints + recalls + qa + manuals
print({"complaints": len(complaints), "recalls": len(recalls), "qa": len(qa), "manuals": len(manuals)})""")

code("""#@title 3. Phase 2 — corpus construction
from autorag.corpus import build_corpus, corpus_stats, save_corpus
chunks = build_corpus(raw)
save_corpus(chunks, "data/corpus.jsonl")
stats = corpus_stats(chunks); stats   # -> Table 4.x in thesis""")

code("""#@title 4. Test set (held-out) + Phase 3a dense retrieval
from autorag.evaluate import build_testset
from autorag.retrieval import Retriever
items, corpus = build_testset(chunks, n_qa=24, n_recall=16)
print("test questions:", len(items), "| retrieval corpus:", len(corpus))
retriever = Retriever(corpus, backend="dense", model_name="sentence-transformers/all-MiniLM-L6-v2")
for c, s in retriever.search("2017 Toyota Camry transmission jerks when shifting", k=3):
    print(round(s,3), c.citation(), c.text[:120])""")

code("""#@title 5. Phase 3b — LLM (Qwen2.5-3B-Instruct, 4-bit). Use 7B if you have >12 GB VRAM
from autorag.generation import HFChat, Generator
llm = HFChat("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
gen = Generator(llm)""")

code("""#@title 6. Phase 4 — self-reflective verification (NLI)
from autorag.verify import NLIVerifier, Verifier
verifier = Verifier(NLIVerifier(threshold=0.3))
from autorag.pipeline import AutoRAG
rag = AutoRAG(retriever, gen, verifier, k=5)
print(rag.answer("My 2016 Honda Civic has a grinding noise from the brakes when cold. What is the likely cause and fix?"))""")

code("""#@title 7. Phase 5 — run all four systems on the test set (≈30–40 min)
from autorag.evaluate import evaluate, plot_summary
# independent, larger judge (Qwen2.5-7B-Instruct, 4-bit) so the generator never grades itself
judge = HFChat("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
df, summary = evaluate(rag, items, judge, verifier, out_dir=OUT, limit=None)
summary""")

code("""#@title 8. Figures + per-source breakdown (paste into thesis Chapter 5)
plot_summary(summary, OUT)
import pandas as pd
by_src = df.groupby(["system","src"])[["correct","faithfulness","retrieval_hit"]].mean().round(3)
by_src.to_csv(OUT + "/summary_by_source.csv"); display(by_src)
from IPython.display import Image, display
display(Image(OUT + "/fig_system_comparison.png")); display(Image(OUT + "/fig_latency.png"))""")

code("""#@title 9. Qualitative examples for the thesis (3 questions × 4 systems)
for q in items[:3]:
    print("="*100); print("Q:", q["question"]); print("GOLD:", q["gold"][:300])
    for s in rag.SYSTEMS:
        a = rag.run(s, q["question"]); print(f"\\n[{s}] {a.text[:500]}")
        if a.verification: print("   verification:", {k:v for k,v in a.verification.items() if k!='claims'})""")

code("""#@title 10. Zip everything for the thesis appendix
!cp data/corpus.jsonl "$OUT"/ && cd "$OUT"/.. && zip -qr /content/thesis_results.zip AutoRAG_results
from google.colab import files; files.download("/content/thesis_results.zip")""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("AutoRAG_Thesis_Experiments.ipynb", "w"), indent=1)
print("notebook written")
