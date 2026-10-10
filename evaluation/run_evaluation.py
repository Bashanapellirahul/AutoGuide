"""Evaluate local retrieval and refusal behavior without paid generation."""
import argparse
import json
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from evaluation.golden_set import GOLDEN_SET
from evaluation.metrics import retrieval_metrics
from rag.config import DATA_DIR
from rag.evidence import EvidenceChecker


def evaluate(semantic=False):
    checker = EvidenceChecker(enable_dense=semantic, rerank=semantic)
    checker.get_retriever()  # Separate model/index startup from query latency.
    rows = []
    for case in GOLDEN_SET:
        start = perf_counter()
        result = checker.check_evidence(case["question"])
        retrieved = [item["chunk"].get("document_id", item["id"]) for item in result["evidence"]]
        row = {**case, "retrieved": retrieved, "reason": result["reason"],
               "latency_ms": round((perf_counter() - start) * 1000, 2)}
        if "relevant" in case:
            row.update(retrieval_metrics(retrieved, case["relevant"]))
        else:
            # No keyword hits and low semantic relevance are both valid
            # insufficient-evidence refusals. Preserve the reason separately.
            accepted = {case["refusal"]}
            if case["refusal"] in {"no_evidence", "low_relevance"}:
                accepted = {"no_evidence", "low_relevance"}
            row["refusal_correct"] = not result["sufficient"] and result["reason"] in accepted
        rows.append(row)
    supported = [r for r in rows if "relevant" in r]
    refused = [r for r in rows if "refusal" in r]
    summary = {key: round(mean(r[key] for r in supported), 4)
               for key in ["hit_at_5", "recall_at_5", "precision_at_5", "mrr_at_5"]}
    summary["refusal_accuracy"] = mean(r["refusal_correct"] for r in refused)
    summary["mean_query_ms"] = round(mean(r["latency_ms"] for r in rows), 2)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "requested_mode": "hybrid_reranked" if semantic else "bm25",
              "warning": checker.get_retriever().warning,
              "scope": "Development set: retrieval and refusals only; no answer faithfulness grading.",
              "summary": summary, "cases": rows}
    output = DATA_DIR / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("hybrid.json" if semantic else "bm25.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detailed report: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true")
    evaluate(parser.parse_args().semantic)
