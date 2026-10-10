"""Hybrid retrieval for AutoGuide AI.

Pipeline:
    question -> BM25 (keyword) + Dense (ChromaDB) -> Reciprocal Rank Fusion
             -> top 20+ candidates -> Cross-Encoder reranker -> top 5 chunks

Chunk format expected from the chunking stage (one dict per chunk):
    {"id": "...", "text": "...", "title": "...", "url": "...", ...metadata}
"""

import json
import hashlib
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rag.config import CHUNKS_DIR, PROCESSED_DIR, DB_PATH, EMBEDDING_MODEL, COLLECTION_NAME
from rag.vector_store import VectorStore, flat_metadata

load_dotenv()

PROCESSED_DIRS = [
    PROCESSED_DIR,
    PROCESSED_DIR / "api",
    PROCESSED_DIR / "features"
]

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DB_PATH = DB_PATH

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "nhtsa_chunks"
)

# How many results each retriever returns before fusion.
PER_RETRIEVER_K = 50

# Proposal requirement: at least 20 candidates before reranking.
CANDIDATE_K = 20

# Final number of chunks sent to the LLM.
FINAL_K = 5

# Standard RRF constant (Cormack et al., 2009).
RRF_K = 60

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "my", "of",
    "on", "or", "the", "to", "what", "when", "where", "which", "who",
    "why", "with", "you", "your"
}


# ---------------------------------------------------------------------------
# Loading chunks
# ---------------------------------------------------------------------------

def normalize_chunk(record):
    """Give every chunk the same keys, whatever stage produced it."""

    chunk_id = (
        record.get("id")
        or record.get("chunk_id")
        or record.get("doc_id")
    )
    if not chunk_id:
        chunk_id = hashlib.sha256((record.get("text", "") + record.get("url", record.get("source_url", ""))).encode()).hexdigest()[:24]

    return {
        **record,
        "id": str(chunk_id),
        "text": record.get("text", ""),
        "title": record.get("title", ""),
        "url": record.get("url") or record.get("source_url", "")
    }


def read_records(file_path):

    if file_path.suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    data = json.loads(file_path.read_text(encoding="utf-8"))

    return data if isinstance(data, list) else [data]


def load_chunks(chunks_dir=CHUNKS_DIR):
    """Load chunks from data/chunks.

    Until the chunking stage is ready, fall back to the processed
    documents and treat each document as a single chunk.
    """

    directories = [Path(chunks_dir)]

    if not any(Path(chunks_dir).glob("*.json*")):
        directories = PROCESSED_DIRS

    chunks = []

    active_file = Path(chunks_dir) / "chunks.jsonl"
    if active_file.exists():
        return [normalize_chunk(record) for record in read_records(active_file)
                if record.get("text", "").strip()]

    for directory in directories:
        for file_path in sorted(directory.glob("*.json*")):
            for record in read_records(file_path):
                chunk = normalize_chunk(record)

                if chunk["text"].strip():
                    chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lowercase word tokens; keeps hyphenated terms like 'f-150'."""

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


class BM25Retriever:

    def __init__(self, chunks):
        self.chunks = chunks

        # Include the title so headings like "Airbag Recall" match.
        corpus = [
            tokenize(f"{chunk['title']} {chunk['text']}")
            for chunk in chunks
        ]
        self.token_sets = [set(tokens) for tokens in corpus]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=PER_RETRIEVER_K):

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # BM25 IDF can be non-positive in a small corpus. Preserve real
        # keyword matches instead of treating them as an empty result.
        scores = [max(float(score), 0.0001 * sum(token in self.token_sets[i]
                                               for token in query_tokens))
                  for i, score in enumerate(scores)]

        ranked = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        return [
            (self.chunks[index]["id"], float(scores[index]))
            for index in ranked[:k]
            if scores[index] > 0
        ]


# ---------------------------------------------------------------------------
# Dense semantic search (ChromaDB)
# ---------------------------------------------------------------------------

def chroma_metadata(chunk):
    """ChromaDB metadata must be flat str/int/float/bool values."""
    return flat_metadata(chunk)


class DenseRetriever(VectorStore):
    """Compatibility interface backed by the shared persistent index."""

    def embed(self, texts):
        return self.encoder.encode(texts)

    def index_missing(self, chunks, batch_size=64):
        return self.sync(chunks, batch_size)


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """Fuse ranked lists of (chunk_id, score).

    RRF score = sum over lists of 1 / (k + rank), rank starting at 1.
    It only uses ranks, so BM25 and cosine scores need no rescaling.
    """

    fused = {}

    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        fused.items(),
        key=lambda item: item[1],
        reverse=True
    )


class HybridRetriever:

    def __init__(self, chunks=None, reranker=None, enable_dense=True):

        self.chunks = chunks if chunks is not None else load_chunks()

        if not self.chunks:
            raise ValueError(
                "No chunks found. Run ingestion/chunking first."
            )

        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}

        self.bm25 = BM25Retriever(self.chunks)
        self.dense = None
        self.warning = None
        if enable_dense:
            try:
                from rag.vector_store import VectorStore
                self.dense = VectorStore(self.chunks)
            except Exception as error:
                self.warning = f"Semantic search unavailable ({type(error).__name__}). Using keyword search."
                import logging
                logging.getLogger(__name__).exception("Semantic index initialization failed")
        self.reranker = reranker

    def candidates(self, query, mode="hybrid", k=CANDIDATE_K, vehicle_filter=None):
        """Return top-k candidates before reranking, with a trace.

        mode: "bm25" (baseline B1), "dense" (B2) or "hybrid" (B3).
        """

        bm25_results = []
        dense_results = []
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode.")
        if mode == "dense" and self.dense is None:
            raise ValueError("Semantic search is not available.")
        if mode == "hybrid" and self.dense is None:
            mode = "bm25"
        search_k = len(self.chunks) if vehicle_filter else PER_RETRIEVER_K

        if mode in ("bm25", "hybrid"):
            bm25_results = self.bm25.search(query, k=search_k)

        if mode in ("dense", "hybrid"):
            dense_results = self.dense.search(query, k=search_k)

        if vehicle_filter:
            def matches(item):
                vehicle = self.by_id.get(item[0], {}).get("vehicle", {})
                return all(str(vehicle.get(key, "")).casefold() == str(value).casefold()
                           for key, value in vehicle_filter.items() if value not in (None, ""))
            bm25_results = [item for item in bm25_results if matches(item)]
            dense_results = [item for item in dense_results if matches(item)]

        if mode == "bm25":
            ranked = bm25_results
        elif mode == "dense":
            ranked = dense_results
        else:
            ranked = reciprocal_rank_fusion(
                [bm25_results, dense_results]
            )

        bm25_lookup = {
            chunk_id: (rank, score)
            for rank, (chunk_id, score) in enumerate(bm25_results, 1)
        }

        dense_lookup = {
            chunk_id: (rank, score)
            for rank, (chunk_id, score) in enumerate(dense_results, 1)
        }

        results = []

        for rank, (chunk_id, score) in enumerate(ranked[:k], start=1):

            chunk = self.by_id.get(chunk_id)

            if chunk is None:
                continue

            bm25_rank, bm25_score = bm25_lookup.get(chunk_id, (None, None))
            dense_rank, dense_score = dense_lookup.get(chunk_id, (None, None))

            results.append({
                "id": chunk_id,
                "title": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
                "chunk": chunk,
                "retrieval_rank": rank,
                "retrieval_score": score,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_score,
                "dense_rank": dense_rank,
                "dense_score": dense_score
            })

        return results

    def retrieve(self, query, top_k=FINAL_K, mode="hybrid",
                 candidate_k=CANDIDATE_K, rerank=True, vehicle_filter=None):
        """Main entry point for the chatbot.

        Returns the top_k chunks, best first. Each result keeps its
        scores before and after reranking for the retrieval trace.
        """

        start = time.perf_counter()

        candidates = self.candidates(query, mode=mode, k=candidate_k, vehicle_filter=vehicle_filter)

        retrieval_ms = (time.perf_counter() - start) * 1000

        if rerank and self.reranker is not None:
            results = self.reranker.rerank(query, candidates, top_k)
        else:
            results = candidates[:top_k]

        total_ms = (time.perf_counter() - start) * 1000

        for result in results:
            result["retrieval_ms"] = round(retrieval_ms, 1)
            result["total_ms"] = round(total_ms, 1)

        return results


# ---------------------------------------------------------------------------
# Simple interface for the chatbot
# ---------------------------------------------------------------------------

_retriever = None


def get_retriever():
    """Build the retriever once and reuse it (models load slowly)."""

    global _retriever

    if _retriever is None:
        from rag.reranker import CrossEncoderReranker

        _retriever = HybridRetriever(reranker=CrossEncoderReranker())

    return _retriever


def retrieve(question, top_k=FINAL_K, mode="hybrid", rerank=True):
    """Return the top chunks for a question.

    Example:
        from rag.retrieval import retrieve
        chunks = retrieve("How does automatic emergency braking work?")
    """

    return get_retriever().retrieve(
        question,
        top_k=top_k,
        mode=mode,
        rerank=rerank
    )


def format_score(value):
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":

    question = " ".join(sys.argv[1:]) or "airbag inflator recall"

    print(f"\nQuestion: {question}\n")

    for result in retrieve(question):
        print(
            f"[{result['retrieval_rank']:>2} -> "
            f"rerank {format_score(result.get('rerank_score'))}] "
            f"bm25 #{result['bm25_rank'] or '-'} "
            f"dense #{result['dense_rank'] or '-'} | "
            f"{result['title']}"
        )
        print(f"     {result['url']}")
