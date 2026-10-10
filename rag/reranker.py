from sentence_transformers import CrossEncoder

from rag.retrieval import HybridRetriever


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self):
        print("Loading cross-encoder model...")

        self.model = CrossEncoder(MODEL_NAME)

        print("Cross-encoder loaded successfully.")

    def rerank(self, query, results, top_k=5):
        """Rerank retrieved documents based on relevance."""

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            updated_result = result.copy()
            updated_result["rerank_score"] = float(score)

            reranked_results.append(updated_result)

        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]


def main():

    retriever = HybridRetriever(strategy="structure")
    reranker = Reranker()

    question = (
        "What are the safety risks associated "
        "with a 2020 Honda Accord airbag recall?"
    )

    vehicle_filter = {
        "make": "HONDA",
        "model": "ACCORD",
        "year": "2020"
    }

    # Retrieve up to 20 candidates before reranking.
    candidates = retriever.search(
        query=question,
        top_k=20,
        vehicle_filter=vehicle_filter
    )

    results = reranker.rerank(
        query=question,
        results=candidates,
        top_k=5
    )

    print("\nQuestion:", question)
    print("Candidates retrieved:", len(candidates))
    print("Reranked results:", len(results))

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nResult {index}")
        print("Title:", metadata.get("title", ""))
        print("Campaign:", metadata.get("campaign_number", ""))
        print("Source:", metadata.get("source_url", ""))
        print("Rerank Score:", round(result["rerank_score"], 4))

        print("\nRetrieved Text:")
        print(result["text"][:500])


if __name__ == "__main__":
    main()