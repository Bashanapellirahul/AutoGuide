from types import SimpleNamespace
import pytest
from rag.chatbot import AutoGuideChatbot
from rag.evidence import EvidenceChecker
from rag.retrieval import HybridRetriever, normalize_chunk


@pytest.fixture
def checker():
    chunks = [normalize_chunk({"id": "h", "document_id": "h", "title": "Honda airbag recall",
               "text": "Vehicle: 2020 Honda Accord. Airbag inflator may rupture. Remedy: Replace the inflator.",
               "url": "https://api.nhtsa.gov/honda", "campaign_number": "TEST-H",
               "vehicle": {"make": "HONDA", "model": "ACCORD", "year": 2020}}),
              normalize_chunk({"id": "f", "document_id": "f", "title": "Ford brake recall",
               "text": "Vehicle: 2020 Ford F-150. Brake hose may leak. Remedy: Replace the hose.",
               "url": "https://api.nhtsa.gov/ford", "vehicle": {"make": "FORD", "model": "F-150", "year": 2020}}),
              normalize_chunk({"id": "a", "document_id": "a", "title": "Automatic emergency braking",
               "text": "Automatic emergency braking applies the brakes when a collision is imminent.",
               "url": "https://www.federalregister.gov/aeb"})]
    return EvidenceChecker(retriever=HybridRetriever(chunks=chunks, enable_dense=False))


def test_evidence_answer_has_correct_vehicle_and_source(checker):
    result = AutoGuideChatbot(checker=checker).answer("2020 Honda Accord airbag recall")
    assert result["status"] == "success"
    assert result["sources"][0]["url"] == "https://api.nhtsa.gov/honda"
    assert "Ford" not in result["answer"]
    assert "inflator" in result["answer"]


@pytest.mark.parametrize("question,reason", [
    ("2020 Tesla Model 3 recalls", "no_evidence"),
    ("2020 Honda Pilot recalls", "no_evidence"),
    ("2020 Honda Accord battery recall", "low_relevance"),
    ("When will my brakes fail?", "unsupported_prediction"),
    ("Check my VIN", "vin_lookup"),
    ("Is my car equipped with AEB?", "feature_fitment"),
    ("How do I cook pasta?", "no_evidence"),
])
def test_unsupported_questions_refused(checker, question, reason):
    result = AutoGuideChatbot(checker=checker).answer(question)
    assert result["status"] == "refused"
    assert result["reason"] == reason
    assert result["sources"] == []


def test_conflicting_vehicle_selection_refused(checker):
    result = AutoGuideChatbot(checker=checker).answer("Ford F-150 brake recall", {"make": "Honda"})
    assert result["reason"] == "vehicle_conflict"


def test_greeting_works_without_data():
    assert AutoGuideChatbot().answer("Hello!")["status"] == "greeting"


def test_feature_acronym(checker):
    result = AutoGuideChatbot(checker=checker).answer("What is AEB?")
    assert result["status"] == "success"
    assert any("federalregister" in s["url"] for s in result["sources"])


def test_feature_explanation_skips_heading_and_uses_later_chunks():
    chunks = [normalize_chunk({"id": "heading", "document_id": "tpms", "title": "Tire pressure monitoring systems",
              "text": "138; Tire pressure monitoring systems.", "source_type": "nhtsa_federal_register",
              "url": "https://www.federalregister.gov/tpms"}),
              normalize_chunk({"id": "body", "document_id": "tpms", "title": "Tire pressure monitoring systems",
              "text": "Direct TPMSs have a pressure sensor in each wheel that transmits pressure information to a receiver. In contrast, indirect TPMSs rely on wheel speed sensors to detect differences in tire pressure. The TPMS is not required to monitor a spare tire when it is installed on the vehicle. The TPMS warns the driver when the pressure in a tire falls below the",
              "source_type": "nhtsa_federal_register", "url": "https://www.federalregister.gov/tpms"})]
    checker = EvidenceChecker(retriever=HybridRetriever(chunks=chunks, enable_dense=False))
    result = AutoGuideChatbot(checker=checker).answer("How do tire pressure monitoring systems work?")
    assert result["status"] == "success"
    assert "pressure sensor in each wheel" in result["answer"]
    assert "indirect TPMSs" in result["answer"]
    assert "138;" not in result["answer"]
    assert "falls below the" not in result["answer"]
    assert "not required" not in result["answer"]
    assert len(result["answer"].split()) < 100
    assert len(result["sources"]) >= 1


def fake_client(answer):
    def create(**kwargs):
        assert "NHTSA Evidence:" in kwargs["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize("answer,status", [
    ("The inflator may rupture [1].", "success"),
    ("The inflator may rupture [99].", "citation_validation_failed"),
    ("The inflator may rupture.", "citation_validation_failed"),
])
def test_llm_citation_numbers(checker, answer, status):
    result = AutoGuideChatbot(checker=checker, client=fake_client(answer)).answer("Honda Accord airbag recall")
    assert result["status"] == status
