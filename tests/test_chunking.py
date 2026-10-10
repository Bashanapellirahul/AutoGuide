import tiktoken
import pytest
from rag.chunking import chunk_document, build_chunks


def test_fixed_chunk_limits_overlap_and_provenance():
    document = {"id": "d1", "text": "brake safety system " * 1000,
                "url": "https://api.nhtsa.gov/recalls", "vehicle": {"make": "Honda"}}
    chunks = chunk_document(document, "fixed")
    encoding = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1
    assert all(c["token_count"] <= 500 for c in chunks)
    assert all(c["url"] == document["url"] and c["vehicle"] == document["vehicle"] for c in chunks)
    assert encoding.encode(chunks[0]["text"])[-100:] == encoding.encode(chunks[1]["text"])[:100]
    assert chunk_document(document, "fixed") == chunks
    assert chunk_document({**document, "text": document["text"] + "updated"}, "fixed")[-1]["id"] != chunks[-1]["id"]


def test_structure_keeps_small_recall_whole():
    doc = {"id": "r1", "text": "Vehicle: Honda\n\nRecall Summary:\nAirbag issue.\n\nRecommended Remedy:\nReplace inflator.",
           "headings": ["Recall Summary", "Recommended Remedy"]}
    chunks = chunk_document(doc, "structure")
    assert len(chunks) == 1
    assert "Airbag issue." in chunks[0]["text"] and "Replace inflator." in chunks[0]["text"]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document({"text": "hello"}, overlap=500)


def test_empty_corpus_rejected(tmp_path):
    with pytest.raises(ValueError, match="No processed documents"):
        build_chunks(processed_dir=tmp_path, output_dir=tmp_path / "chunks")
