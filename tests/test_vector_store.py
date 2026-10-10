import pytest
from rag.vector_store import VectorStore


class FakeEncoder:
    model_name = "test-encoder"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [[1.0, 0.0, 0.0] if "airbag" in t else [0.0, 1.0, 0.0] for t in texts]


def test_chroma_updates_and_removes_stale_documents(tmp_path):
    chunks = [{"id": "a", "title": "airbag", "text": "airbag risk", "url": "https://api.nhtsa.gov/a"},
              {"id": "b", "title": "brake", "text": "brake leak", "url": "https://api.nhtsa.gov/b"}]
    store = VectorStore(chunks, db_path=tmp_path, encoder=FakeEncoder(), collection_name="test_chunks")
    assert store.collection.count() == 2
    assert store.search("airbag")[0][0] == "a"
    assert store.sync(chunks) == 0
    changed = [{**chunks[0], "text": "airbag corrected risk"}]
    assert store.sync(changed) == 1
    assert store.collection.count() == 1
    assert store.collection.get(ids=["a"])["documents"] == ["airbag corrected risk"]
    reopened = VectorStore(changed, db_path=tmp_path, encoder=FakeEncoder(), collection_name="test_chunks")
    assert reopened.collection.count() == 1


def test_metadata_refresh_reuses_vectors(tmp_path):
    encoder = FakeEncoder()
    chunk = {"id": "a", "title": "airbag", "text": "airbag risk", "retrieved_at": "old"}
    store = VectorStore([chunk], db_path=tmp_path, encoder=encoder, collection_name="metadata_test")
    assert encoder.calls == 1
    assert store.sync([{**chunk, "retrieved_at": "new"}]) == 1
    assert encoder.calls == 1
    assert store.collection.get(ids=["a"])["metadatas"][0]["retrieved_at"] == "new"
