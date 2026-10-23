"""
AutoRAG — Multi-Source Retrieval-Augmented Generation for Automotive Fault Diagnosis.

Implements the five-phase methodology of the research proposal:
  Phase 1  data        -> collect four public sources (complaints, Q&A, recalls, manuals)
  Phase 2  corpus      -> clean, chunk, enrich with metadata
  Phase 3  retrieval   -> dense (Sentence-Transformers + FAISS) metadata-aware retrieval
           generation  -> grounded, source-attributed LLM generation
  Phase 4  verify      -> self-reflective claim-level grounding check (NLI)
  Phase 5  evaluate    -> LLM-only / vanilla-RAG / multi-source-RAG / +verification
"""
__version__ = "0.1.0"
