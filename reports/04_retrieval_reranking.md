# Retrieval and reranking

BM25 searches title and content. The semantic index uses normalized
all-MiniLM-L6-v2 vectors and Chroma cosine distance. Longer chunks pool short
MiniLM windows to avoid losing text beyond its input limit.

Hybrid search fuses keyword and semantic ranks using RRF with k=60. Up to
30 candidates are considered by evidence checking, with up to five passed
on after cross-encoder reranking. Tiny or filtered corpora may contain fewer
than 20 candidates; the code does not invent extra candidates.

Vehicle filtering happens before candidate truncation. Explicit question
vehicles conflicting with sidebar selection are refused. Missing semantic
models fall back to keyword search with a visible warning; a missing reranker
falls back to fusion ranks with a warning.

The development evaluation is available via `python -m
evaluation.run_evaluation`, optionally with `--semantic`. These development
metrics do not establish claim-level faithfulness or final assignment quality.
