import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.vector_store import VectorStore


CHUNKS_FILE = Path("data/chunks/recall_chunks.json")


class HybridRetriever:

    def __init__(self, strategy="structure"):
        self.strategy = strategy
        self.vector_store = VectorStore()

        with CHUNKS_FILE.open("r", encoding="utf-8") as file:
            all_chunks = json.load(file)

        self.chunks = [
            chunk
            for chunk in all_chunks
            if chunk["strategy"] == strategy
        ]

        self.chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"Hybrid retriever ready: {len(self.chunks)} chunks")

    def tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def matches_vehicle(self, chunk, vehicle_filter):
        if not vehicle_filter:
            return True

        vehicle = chunk.get("vehicle", {})

        for key, expected in vehicle_filter.items():
            actual = str(vehicle.get(key, "")).strip().lower()

            if actual != str(expected).strip().lower():
                return False

        return True

    def bm25_search(self, query, top_k=20, vehicle_filter=None):
        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for chunk, score in ranked:
            if not self.matches_vehicle(chunk, vehicle_filter):
                continue

            if score <= 0:
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk,
                "score": float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def dense_search(self, query, top_k=20, vehicle_filter=None):
        collection = self.vector_store.get_collection(self.strategy)

        if collection.count() == 0:
            return []

        query_embedding = (
            self.vector_store.embedding_model
            .generate_embeddings([query])[0]
        )

        where = None

        if vehicle_filter:
            conditions = [
                {key: str(value)}
                for key, value in vehicle_filter.items()
            ]

            where = (
                conditions[0]
                if len(conditions) == 1
                else {"$and": conditions}
            )

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            query_args["where"] = where

        results = collection.query(**query_args)

        formatted = []

        for index, chunk_id in enumerate(results["ids"][0]):
            formatted.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return formatted

    def reciprocal_rank_fusion(self, dense_results, bm25_results, k=60):
        fused_scores = {}
        result_lookup = {}

        for results in [dense_results, bm25_results]:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0)
                    + 1 / (k + rank)
                )

                result_lookup[chunk_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )

        final_results = []

        for chunk_id in ranked_ids:
            result = result_lookup[chunk_id].copy()
            result["rrf_score"] = fused_scores[chunk_id]
            final_results.append(result)

        return final_results

    def search(self, query, top_k=5, vehicle_filter=None):
        dense_results = self.dense_search(
            query,
            top_k=20,
            vehicle_filter=vehicle_filter
        )

        bm25_results = self.bm25_search(
            query,
            top_k=20,
            vehicle_filter=vehicle_filter
        )

        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )

        return fused_results[:top_k]


def main():
    retriever = HybridRetriever(strategy="structure")

    question = (
        "What are the safety risks associated "
        "with a 2020 Honda Accord airbag recall?"
    )

    vehicle_filter = {
        "make": "HONDA",
        "model": "ACCORD",
        "year": "2020"
    }

    results = retriever.search(
        query=question,
        top_k=5,
        vehicle_filter=vehicle_filter
    )

    print("\nQuestion:", question)
    print("Retrieved results:", len(results))

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult {index}")
        print("Title:", metadata.get("title", ""))
        print("Campaign:", metadata.get("campaign_number", ""))
        print("Source:", metadata.get("source_url", ""))
        print("RRF Score:", round(result["rrf_score"], 5))
        print("Text:", result["text"][:400])


if __name__ == "__main__":
    main()