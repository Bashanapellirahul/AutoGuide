from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully.")

    def generate_embeddings(self, texts):
        """Generate normalized embeddings for a list of texts."""

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()


if __name__ == "__main__":

    model = EmbeddingModel()

    texts = [
        "What is automatic emergency braking?",
        "What are the safety risks of defective airbags?"
    ]

    embeddings = model.generate_embeddings(texts)

    print("Total embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))