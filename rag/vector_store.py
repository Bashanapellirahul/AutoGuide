"""Persistent Chroma index that reconciles changed and removed chunks."""
import hashlib
import json
from rag.config import DB_PATH, COLLECTION_NAME
from rag.embeddings import EmbeddingEncoder


def flat_metadata(chunk):
    metadata = {}
    for key, value in chunk.items():
        if key in {"id", "text"}:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    metadata[f"{key}_{sub_key}"] = sub_value
    return metadata


class VectorStore:
    def __init__(self, chunks, db_path=DB_PATH, encoder=None, collection_name=COLLECTION_NAME):
        import chromadb
        self.encoder = encoder or EmbeddingEncoder()
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine", "embedding_model": self.encoder.model_name})
        info = self.collection.metadata or {}
        model = self.encoder.model_name
        if info.get("embedding_model") not in {None, model}:
            raise ValueError("Embedding model changed. Use a new CHROMA_COLLECTION.")
        if info.get("embedding_model") != model:
            # Chroma rejects modify() when hnsw:space is present, even when
            # unchanged. The distance metric is retained by the collection.
            self.collection.modify(metadata={k: v for k, v in {**info, "embedding_model": model}.items()
                                             if not k.startswith("hnsw:")})
        self.sync(chunks)

    @staticmethod
    def content_hash(chunk):
        return hashlib.sha256(json.dumps(chunk, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def sync(self, chunks, batch_size=64):
        existing = self.collection.get(include=["metadatas", "documents", "embeddings"])
        hashes = {key: (meta or {}).get("indexed_hash")
                  for key, meta in zip(existing["ids"], existing["metadatas"])}
        stored = {key: (text, meta or {}, vector) for key, text, meta, vector in
                  zip(existing["ids"], existing["documents"], existing["metadatas"], existing["embeddings"])}
        ids = {c["id"] for c in chunks}
        obsolete = [key for key in hashes if key not in ids]
        if obsolete:
            self.collection.delete(ids=obsolete)
        changed = [c for c in chunks if hashes.get(c["id"]) != self.content_hash(c)]
        for start in range(0, len(changed), batch_size):
            batch = changed[start:start + batch_size]
            # A new retrieval timestamp or source metadata does not require
            # another model pass when the embedded title/text are unchanged.
            vectors = [None] * len(batch)
            positions, texts = [], []
            for position, chunk in enumerate(batch):
                previous = stored.get(chunk["id"])
                if previous and previous[0] == chunk["text"] and previous[1].get("title", "") == chunk.get("title", ""):
                    vectors[position] = list(previous[2])
                else:
                    positions.append(position)
                    texts.append(f"{chunk.get('title', '')}\n{chunk['text']}")
            if texts:
                for position, vector in zip(positions, self.encoder.encode(texts)):
                    vectors[position] = vector
            self.collection.upsert(
                ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                embeddings=vectors,
                metadatas=[{**flat_metadata(c), "indexed_hash": self.content_hash(c)} for c in batch])
        return len(changed)

    def search(self, query, k=50):
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(query_embeddings=self.encoder.encode([query]),
                                       n_results=min(k, count), include=["distances"])
        return [(key, 1.0 - float(distance)) for key, distance
                in zip(result["ids"][0], result["distances"][0])]


if __name__ == "__main__":
    from rag.retrieval import load_chunks
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks. Run python ingest.py first.")
    store = VectorStore(chunks)
    print(f"ChromaDB ready: {store.collection.count()} chunks.")
