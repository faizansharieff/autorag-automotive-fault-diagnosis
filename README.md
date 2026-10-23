# AutoRAG — Multi-Source RAG Framework for Automotive Fault Diagnosis

Prototype for the Master's thesis *"A Multi-Source Retrieval-Augmented Generation Framework for
Automotive Fault Diagnosis"* (Faizan Sharieff, 2026). Implements the five-phase methodology of
the research proposal and the evaluation protocol of Table 7.2.

```
autorag/
  data.py        Phase 1  – NHTSA complaints & recalls (live API), diagnostic Q&A CSV, manuals (PDF/txt)
  corpus.py      Phase 2  – cleaning, de-dup, chunking (220 words / 40 overlap), metadata
  retrieval.py   Phase 3a – Sentence-Transformers (MiniLM) + FAISS, metadata-aware re-ranking
  generation.py  Phase 3b – grounded generation with [n] citations; HF / API / stub backends
  verify.py      Phase 4  – self-reflective claim-level NLI verification, revise/hedge policy
  pipeline.py    the four systems: llm_only · vanilla_rag · multisource_rag · multisource_rag_verified
  evaluate.py    Phase 5  – held-out test set, LLM-judge accuracy/relevance, ROUGE-L,
                            faithfulness (NLI), citation rate, retrieval hit@k, latency, plots
tests/smoke_test.py          offline end-to-end check (no GPU / internet)
AutoRAG_Thesis_Experiments.ipynb   Colab notebook that runs the whole experiment
```

## Results (run 2, 40 held-out questions, 160 answers)

| System | Correct | Relevance | ROUGE-L | Faithfulness | Citation rate | Retrieval hit | Latency (s) |
|---|---|---|---|---|---|---|---|
| S0 LLM only | 0.188 | 0.417 | 0.103 | – | – | – | 11.1 |
| S1 Vanilla RAG | 0.125 | 0.345 | 0.099 | 0.222 | 0.244 | 0.000 | 7.0 |
| S2 Multi-source RAG | 0.225 | 0.398 | 0.190 | 0.290 | 0.506 | 0.175 | 7.6 |
| S3 Multi-source + verification | 0.262 | 0.421 | 0.154 | 0.396 | 0.728 | 0.175 | 8.0 |

On the 16 recall look-up questions, multi-source retrieval raised correctness from 0.06 (vanilla) / 0.13 (LLM only) to 0.41 (Wilcoxon p < 0.05) and verification raised faithfulness from 0.45 to 0.85 (p = 0.01). Per-question outputs: `results/run2_per_question.csv`; the earlier strict-verifier pilot: `results/run1_per_question.csv`.

## Run on Google Colab (recommended)
1. Open `AutoRAG_Thesis_Experiments.ipynb` in Colab, set runtime to **T4 GPU**.
2. Run all cells: the notebook clones this repo, mounts Google Drive and checkpoints after every question.
3. Optional: put your own Q&A CSV at `data/qa.csv` or service-manual PDFs in `data/manuals/`
   (otherwise public Hugging Face Q&A and the OBD-II DTC reference are used).
4. Outputs land in Drive `AutoRAG_results/`:
   `summary.csv` (Table 5.1), `summary_by_source.csv`, `per_question.csv`, `fig_system_comparison.png`, `fig_latency.png`.

## Run locally (CPU, quick check)
```
pip install -r requirements.txt
python -I tests/smoke_test.py
```

## Design choices (cite in Chapter 4)
* **Embeddings** all-MiniLM-L6-v2 (384-d, cosine via inner product on normalised vectors, FAISS IndexFlatIP).
* **Metadata-aware retrieval** query is parsed for make / year / component; matching chunks get a small score boost — this is how the multi-source system beats the single-source baseline on vehicle-specific questions.
* **Generator** Qwen2.5-3B-Instruct (4-bit NF4, fits T4), max 160 new tokens, greedy. Prompt restricts the model to the numbered evidence and requires `[n]` citations (FiD-style evidence aggregation).
* **Verification** answer → sentence-level claims → `cross-encoder/nli-deberta-v3-base` entailment against cited passage first, then all passages; threshold 0.3; unsupported claims removed; if none survive, the draft is returned with a low-confidence warning.
* **Baselines** S1 vanilla RAG = complaints only, no metadata, no verification — isolates the contribution of (a) multi-source corpus, (b) verification layer.
* **Judge** independent Qwen2.5-7B-Instruct (4-bit), JSON 0/0.5/1 correctness and 1–5 relevance.
* **Data** NHTSA complaints + recalls (live API, 23 vehicles), Hugging Face `Ichlibitiche/mechanicdb-obd2-repair-sample`, `Epitech/obd-codes-fine-tune`, `CodesList/obd2-codes`.

## Mapping to research questions
| RQ | Evidence produced |
|---|---|
| RQ1 integration | `corpus_stats()` table, schema in `corpus.py`, metadata fields |
| RQ2 accuracy/relevance/explainability | `summary.csv`: correct, relevance, citation_rate — S2 vs S0/S1 |
| RQ3 hallucination/faithfulness | `summary.csv`: faithfulness, hallucination_rate — S3 vs S2 |
| RQ4 evaluation protocol | `evaluate.py` metric definitions + latency trade-off figure |
