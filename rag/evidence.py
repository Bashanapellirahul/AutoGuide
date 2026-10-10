"""Conservative relevance checks and explicit vehicle scoping."""
import re
from rag.retrieval import HybridRetriever, tokenize

ALIASES = {
    "aeb": "automatic emergency braking", "abs": "anti lock brake braking",
    "esc": "electronic stability control", "tpms": "tire pressure monitoring",
    "airbags": "airbag air bags", "airbag": "airbag air bags", "brakes": "brake braking",
    "camera": "camera rear visibility", "blindspot": "blind spot",
}
GENERIC = {"vehicle", "vehicles", "car", "cars", "safety", "safe", "feature", "features",
           "recall", "recalls", "tell", "about", "explain", "please", "does", "work", "works",
           "associated", "risks", "risk", "information", "find", "have", "has", "there",
           "affect", "affected", "available", "list", "show", "known", "any", "me"}
MAKES = {"honda", "toyota", "ford", "tesla", "chevrolet", "bmw", "audi", "nissan", "kia",
         "hyundai", "mazda", "subaru", "volkswagen", "mercedes", "jeep", "dodge", "volvo",
         "lexus", "gmc", "ram", "buick", "chrysler", "acura", "mitsubishi", "porsche"}


def expand_query(question):
    additions = [value for key, value in ALIASES.items()
                 if re.search(r"\b" + re.escape(key) + r"\b", question.casefold())]
    for acronym in ("tpms", "aeb", "esc"):
        if ALIASES[acronym] in question.casefold():
            additions.append(acronym)
    if "tire pressure monitoring" in question.casefold() or re.search(r"\btpms\b", question, re.I):
        additions.append("direct indirect pressure sensors wheel speed under-inflated alert driver")
    return question + " " + " ".join(additions)


def infer_vehicle(question, chunks):
    lowered = question.casefold()
    selected = {}
    year = re.search(r"\b(?:19|20)\d{2}\b", lowered)
    if year:
        selected["year"] = year.group()
    known_makes = MAKES | {str(c.get("vehicle", {}).get("make", "")).casefold() for c in chunks} - {""}
    for make in sorted(known_makes, key=len, reverse=True):
        if re.search(r"\b" + re.escape(make) + r"\b", lowered):
            selected["make"] = make
            break
    models = {str(c.get("vehicle", {}).get("model", "")).casefold() for c in chunks} - {""}
    for model in sorted(models, key=len, reverse=True):
        if re.search(r"\b" + re.escape(model) + r"\b", lowered):
            selected["model"] = model
            break
    if "make" in selected and "model" not in selected:
        after = re.search(r"\b" + re.escape(selected["make"]) + r"\s+([a-z][a-z0-9-]*)", lowered)
        if after and after.group(1) not in GENERIC | {"and", "or", "in", "is", "the", "with", "for"}:
            selected["model"] = after.group(1)
    if set(selected) == {"year"} and not re.search(r"\brecalls?\b", lowered):
        return {}
    return selected


class EvidenceChecker:
    def __init__(self, retriever=None, enable_dense=False, rerank=False):
        self.retriever = retriever
        self.enable_dense = enable_dense
        self.use_reranker = rerank

    def get_retriever(self):
        if self.retriever is None:
            self.retriever = HybridRetriever(enable_dense=self.enable_dense)
            if self.use_reranker and self.retriever.dense is not None:
                try:
                    from rag.reranker import CrossEncoderReranker
                    self.retriever.reranker = CrossEncoderReranker()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Reranker unavailable")
                    self.retriever.warning = "Reranker unavailable; using fused search ranking."
        return self.retriever

    def check_evidence(self, question, vehicle_filter=None):
        def refusal(reason):
            return {"sufficient": False, "reason": reason, "evidence": []}
        if re.search(r"\b(predict|will.*fail|when.*fail|when.*break|guarantee)\b", question, re.I):
            return refusal("unsupported_prediction")
        if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", question, re.I) or re.search(r"\bvin\b", question, re.I):
            return refusal("vin_lookup")
        if re.search(r"\b(equipped|installed|come with|comes with|has aeb|have aeb)\b", question, re.I):
            return refusal("feature_fitment")
        if re.search(r"\b(current law|legal requirement|compliance|effective date)\b", question, re.I):
            return refusal("current_law")
        try:
            retriever = self.get_retriever()
        except ValueError:
            return refusal("no_data")
        inferred = infer_vehicle(question, retriever.chunks)
        supplied = {k: v for k, v in (vehicle_filter or {}).items() if v not in (None, "")}
        if any(k in inferred and str(inferred[k]).casefold() != str(v).casefold() for k, v in supplied.items()):
            return refusal("vehicle_conflict")
        scope = {**inferred, **supplied}
        expanded = expand_query(question)
        candidates = retriever.candidates(expanded, mode="hybrid", k=30, vehicle_filter=scope)
        if not candidates:
            return refusal("no_evidence")
        subjects = set(tokenize(expanded)) - GENERIC - set(tokenize(" ".join(str(v) for v in scope.values())))
        relevant = []
        for candidate in candidates:
            words = set(tokenize(candidate["text"]))
            matched = subjects & words
            # Fusion scores are not probabilities: a year/make alone cannot
            # support a component claim. Require actual subject overlap.
            if subjects and (not matched or len(matched) / len(subjects) < 0.2):
                continue
            candidate["subject_overlap"] = len(matched) / max(1, len(subjects))
            chunk = candidate["chunk"]
            if not chunk.get("url"):
                continue
            candidate["metadata"] = {**chunk, "source_url": chunk["url"]}
            relevant.append(candidate)
        if not relevant:
            return refusal("low_relevance")
        if retriever.reranker is not None:
            rerank_query = expanded
            if not scope and re.search(r"\b(how|what is|what are|explain)\b", question, re.I):
                rerank_query += " Explain the purpose of the system and how it operates."
            relevant = retriever.reranker.rerank(rerank_query, relevant, top_k=5)
        else:
            relevant = relevant[:5]
        return {"sufficient": True, "reason": "relevant_evidence", "evidence": relevant,
                "vehicle_filter": scope, "warning": retriever.warning}
