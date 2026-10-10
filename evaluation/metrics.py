"""Document-level retrieval metrics; repeated chunks count once."""
def retrieval_metrics(retrieved, relevant, k=5):
    unique = list(dict.fromkeys(retrieved))[:k]
    relevant = set(relevant)
    hits = sum(item in relevant for item in unique)
    rank = next((i for i, item in enumerate(unique, 1) if item in relevant), None)
    return {"hit_at_5": float(hits > 0), "recall_at_5": hits / max(1, len(relevant)),
            "precision_at_5": hits / max(1, len(unique)), "mrr_at_5": 1 / rank if rank else 0.0}
