"""Collect NHTSA publications through the official Federal Register API.

These dated rule publications explain features; they are not a live legal
compliance database or evidence that a particular vehicle has a feature.
"""
import hashlib
import json
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from ingestion.cleaner import clean_text
from rag.config import DATA_DIR, PROCESSED_DIR

API = "https://www.federalregister.gov/api/v1"
TOPICS = {
    "automatic emergency braking": "2024-09054",
    "seat belt reminders": "2024-30340",
    "electronic stability control": "2015-14127",
}
SEARCH_TOPICS = ["tire pressure monitoring", "rear visibility", "air bags"]


def collect_features(topics=None, include_search=True):
    session = requests.Session()
    session.headers.update({"User-Agent": "AutoGuideAI-AcademicResearch/1.0", "Accept": "application/json"})
    raw_dir = DATA_DIR / "raw" / "features"
    output_dir = PROCESSED_DIR / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = dict(TOPICS if topics is None else topics)
    failures = []
    for topic in (SEARCH_TOPICS if include_search else []):
        try:
            response = session.get(f"{API}/documents.json", params={
                "conditions[term]": f'"{topic}"',
                "conditions[agencies][]": "national-highway-traffic-safety-administration",
                "conditions[type][]": "RULE", "per_page": 20, "order": "relevance"}, timeout=30)
            response.raise_for_status()
            results = response.json().get("results", [])
            # Prefer a feature standard, rather than a manufacturer petition.
            match = next((r for r in results if not r.get("document_number", "").startswith(("C1-", "C2-"))
                          and "Federal Motor Vehicle Safety Standards" in r.get("title", "")
                          and topic.casefold() in r.get("title", "").casefold()), None)
            if match:
                selected[topic] = match["document_number"]
            else:
                failures.append(f"No suitable publication found for {topic}")
        except (requests.RequestException, ValueError) as error:
            failures.append(f"{topic}: {type(error).__name__}")
        time.sleep(1)
    saved = []
    for topic, number in selected.items():
        try:
            response = session.get(f"{API}/documents/{number}.json", timeout=30)
            response.raise_for_status()
            document = response.json()
            if not any(a.get("slug") == "national-highway-traffic-safety-administration"
                       for a in document.get("agencies", [])):
                raise ValueError("Publication is not attributed to NHTSA")
            text_url = document.get("raw_text_url")
            if not text_url or not text_url.startswith("https://www.federalregister.gov/"):
                raise ValueError("Missing official full-text URL")
            time.sleep(1)
            text_response = session.get(text_url, timeout=45)
            text_response.raise_for_status()
            text = clean_text(BeautifulSoup(text_response.content, "html.parser").get_text("\n", strip=True))
            if len(text) < 200:
                raise ValueError("Publication text is empty")
            timestamp = datetime.now(timezone.utc).isoformat()
            digest = hashlib.sha256(text_response.content).hexdigest()
            (raw_dir / f"{number}.html").write_bytes(text_response.content)
            (raw_dir / f"{number}.json").write_text(json.dumps({
                **document, "retrieved_at": timestamp, "content_hash": digest}, indent=2), encoding="utf-8")
            normalized = {"id": f"fr-{number}", "title": document["title"],
                          "text": text, "url": document["html_url"], "topic": topic,
                          "source_type": "nhtsa_federal_register", "publication_date": document["publication_date"],
                          "retrieved_at": timestamp, "content_hash": digest, "headings": []}
            (output_dir / f"fr-{number}.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            # Corrections are not suitable stand-alone feature explanations.
            for old_path in output_dir.glob("fr-C*-*.json"):
                old_path.unlink()
            saved.append(normalized)
            print(f"Saved feature publication: {topic} ({number})")
        except (requests.RequestException, ValueError, KeyError) as error:
            failures.append(f"{number}: {type(error).__name__}")
        time.sleep(1)
    return {"saved": len(saved), "failures": failures}


if __name__ == "__main__":
    print(collect_features())
