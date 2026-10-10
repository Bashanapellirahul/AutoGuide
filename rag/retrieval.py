"""Hybrid retrieval for AutoGuide AI.

Pipeline:
    question -> BM25 (keyword) + Dense (ChromaDB) -> Reciprocal Rank Fusion
             -> top 20+ candidates -> Cross-Encoder reranker -> top 5 chunks

Chunk format expected from the chunking stage (one dict per chunk):
    {"id": "...", "text": "...", "title": "...", "url": "...", ...metadata}
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

CHUNKS_DIR = Path("data/chunks")
PROCESSED_DIRS = [
    Path("data/processed"),
    Path("data/processed/api")
]

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "data/chroma_db"
)

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

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=PER_RETRIEVER_K):

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

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

    metadata = {}

    for key, value in chunk.items():

        if key in ("id", "text"):
            continue

        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    metadata[f"{key}_{sub_key}"] = sub_value

    return metadata


class DenseRetriever:

    def __init__(self, chunks, db_path=CHROMA_DB_PATH):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=str(db_path))

        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self.index_missing(chunks)

    def embed(self, texts):
        # L2-normalized so cosine similarity == dot product.
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

    def index_missing(self, chunks, batch_size=64):
        """Add chunks that are not in ChromaDB yet.

        If the chunking stage already built the collection, this
        does nothing.
        """

        existing = set(self.collection.get(include=[])["ids"])

        missing = [
            chunk
            for chunk in chunks
            if chunk["id"] not in existing
        ]

        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]

            self.collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                embeddings=self.embed([
                    f"{chunk['title']}\n{chunk['text']}"
                    for chunk in batch
                ]),
                metadatas=[chroma_metadata(chunk) for chunk in batch]
            )

        if missing:
            print(f"Indexed {len(missing)} new chunks in ChromaDB.")

    def search(self, query, k=PER_RETRIEVER_K):

        count = self.collection.count()

        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=self.embed([query]),
            n_results=min(k, count),
            include=["distances"]
        )

        # Chroma returns cosine distance; similarity = 1 - distance.
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(
                result["ids"][0],
                result["distances"][0]
            )
        ]


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

    def __init__(self, chunks=None, reranker=None):

        self.chunks = chunks if chunks is not None else load_chunks()

        if not self.chunks:
            raise ValueError(
                "No chunks found. Run ingestion/chunking first."
            )

        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}

        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks)
        self.reranker = reranker

    def candidates(self, query, mode="hybrid", k=CANDIDATE_K):
        """Return top-k candidates before reranking, with a trace.

        mode: "bm25" (baseline B1), "dense" (B2) or "hybrid" (B3).
        """

        bm25_results = []
        dense_results = []

        if mode in ("bm25", "hybrid"):
            bm25_results = self.bm25.search(query)

        if mode in ("dense", "hybrid"):
            dense_results = self.dense.search(query)

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
                 candidate_k=CANDIDATE_K, rerank=True):
        """Main entry point for the chatbot.

        Returns the top_k chunks, best first. Each result keeps its
        scores before and after reranking for the retrieval trace.
        """

        start = time.perf_counter()

        candidates = self.candidates(query, mode=mode, k=candidate_k)

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
