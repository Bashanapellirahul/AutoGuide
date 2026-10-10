from evaluation.metrics import retrieval_metrics


def test_document_metrics_deduplicate_chunks():
    result = retrieval_metrics(["wrong", "a", "a", "b"], ["a", "b"])
    assert result["recall_at_5"] == 1
    assert result["mrr_at_5"] == 0.5
    assert result["precision_at_5"] == 2 / 3


def test_empty_retrieval():
    assert retrieval_metrics([], ["a"])["hit_at_5"] == 0
