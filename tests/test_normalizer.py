import json
from ingestion import api_normalizer


def test_refresh_removes_obsolete_recall_for_same_vehicle(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    monkeypatch.setattr(api_normalizer, "RAW_API_DIR", raw)
    monkeypatch.setattr(api_normalizer, "PROCESSED_API_DIR", processed)
    vehicle = {"make": "Honda", "model": "Accord", "year": 2020}
    stale = {"id": "old", "vehicle": {"make": "HONDA", "model": "ACCORD", "year": "2020"}}
    other = {"id": "other", "vehicle": {"make": "FORD", "model": "F-150", "year": 2020}}
    (processed / "old.json").write_text(json.dumps(stale))
    (processed / "other.json").write_text(json.dumps(other))
    (raw / "honda.json").write_text(json.dumps({"vehicle": vehicle, "source_url": "https://api.nhtsa.gov/recalls",
        "retrieved_at": "2026-10-10", "records": [{"NHTSACampaignNumber": "TEST", "Summary": "Airbag issue"}]}))
    api_normalizer.process_api_documents()
    assert not (processed / "old.json").exists()
    assert (processed / "other.json").exists()
    normalized = [json.loads(p.read_text()) for p in processed.glob("*.json") if p.name != "other.json"]
    assert len(normalized) == 1
    assert normalized[0]["retrieved_at"] == "2026-10-10"
