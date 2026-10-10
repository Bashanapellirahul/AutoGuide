import json
from pathlib import Path

import chromadb

from rag.embeddings import EmbeddingModel


CHUNKS_FILE = Path("data/chunks/recall_chunks.json")
DB_PATH = "data/chroma_db"


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=DB_PATH
        )

        self.embedding_model = EmbeddingModel()

    def get_collection(self, strategy):

        collection_name = f"autoguide_{strategy}"

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def prepare_metadata(self, chunk):

        vehicle = chunk.get("vehicle", {})

        return {
            "document_id": chunk.get("document_id", ""),
            "strategy": chunk.get("strategy", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "title": chunk.get("title", ""),
            "source_url": chunk.get("source_url", ""),
            "campaign_number": chunk.get(
                "campaign_number", ""
            ),
            "make": str(vehicle.get("make", "")),
            "model": str(vehicle.get("model", "")),
            "year": str(vehicle.get("year", ""))
        }

    def index_chunks(self, chunks, strategy):

        selected_chunks = [
            chunk for chunk in chunks
            if chunk["strategy"] == strategy
        ]

        if not selected_chunks:
            print(f"No {strategy} chunks found.")
            return

        collection = self.get_collection(strategy)

        texts = [
            chunk["text"]
            for chunk in selected_chunks
        ]

        embeddings = (
            self.embedding_model.generate_embeddings(texts)
        )

        ids = [
            chunk["chunk_id"]
            for chunk in selected_chunks
        ]

        metadatas = [
            self.prepare_metadata(chunk)
            for chunk in selected_chunks
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"{strategy} collection indexed: "
            f"{collection.count()} chunks"
        )

    def search(self, query, strategy="structure", top_k=5):

        collection = self.get_collection(strategy)

        if collection.count() == 0:
            print("Collection is empty.")
            return []

        query_embedding = (
            self.embedding_model.generate_embeddings([query])[0]
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []

        for index, chunk_id in enumerate(results["ids"][0]):

            formatted_results.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            })

        return formatted_results


def build_vector_database():

    if not CHUNKS_FILE.exists():
        print("Chunk file not found.")
        return

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    vector_store = VectorStore()

    vector_store.index_chunks(chunks, "fixed")
    vector_store.index_chunks(chunks, "structure")

    print("\nChromaDB indexing completed successfully.")


if __name__ == "__main__":
    build_vector_database()