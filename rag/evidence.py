import re

from rag.retrieval import HybridRetriever
from rag.reranker import Reranker


MIN_RERANK_SCORE = 0.0

UNSUPPORTED_PATTERNS = [
    r"\bdefinitely\b.*\b(tomorrow|next week|next month)\b",
    r"\bguarantee\b",
    r"\bexactly when\b.*\b(fail|break)\b",
    r"\bwill\b.*\bdefinitely\b.*\b(fail|break)\b",
]


class EvidenceChecker:

    def __init__(self):
        self.retriever = HybridRetriever(strategy="structure")
        self.reranker = Reranker()

    def is_unsupported_prediction(self, question):
        question = question.lower()

        return any(
            re.search(pattern, question)
            for pattern in UNSUPPORTED_PATTERNS
        )

    def check_evidence(self, question, vehicle_filter=None):

        if self.is_unsupported_prediction(question):
            return {
                "sufficient": False,
                "reason": "unsupported_prediction",
                "evidence": []
            }

        candidates = self.retriever.search(
            query=question,
            top_k=20,
            vehicle_filter=vehicle_filter
        )

        if not candidates:
            return {
                "sufficient": False,
                "reason": "no_evidence",
                "evidence": []
            }

        reranked = self.reranker.rerank(
            query=question,
            results=candidates,
            top_k=5
        )

        relevant = [
            result
            for result in reranked
            if result["rerank_score"] >= MIN_RERANK_SCORE
        ]

        if not relevant:
            return {
                "sufficient": False,
                "reason": "low_relevance",
                "evidence": []
            }

        return {
            "sufficient": True,
            "reason": "relevant_evidence_found",
            "evidence": relevant
        }


def main():

    checker = EvidenceChecker()

    vehicle_filter = {
        "make": "HONDA",
        "model": "ACCORD",
        "year": "2020"
    }

    questions = [
        "What are the safety risks associated with a 2020 Honda Accord airbag recall?",
        "Will my 2020 Honda Accord definitely have an airbag failure tomorrow?",
        "What is the recall status of my specific vehicle?"
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("Question:", question)

        result = checker.check_evidence(
            question,
            vehicle_filter=vehicle_filter
        )

        print("Evidence sufficient:", result["sufficient"])
        print("Reason:", result["reason"])
        print("Evidence count:", len(result["evidence"]))


if __name__ == "__main__":
    main()