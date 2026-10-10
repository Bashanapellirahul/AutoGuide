"""Local sentence-transformer embeddings; no OpenAI key required."""
from functools import lru_cache
from rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def get_model(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        return SentenceTransformer(model_name)


class EmbeddingEncoder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = get_model(model_name)
        self.model_name = model_name

    def encode(self, texts):
        # Pool shorter windows so a 500-token chunk is not silently truncated
        # by MiniLM's 256-wordpiece input limit.
        import numpy as np
        tokenizer = self.model.tokenizer
        max_tokens = self.model.max_seq_length - 2
        windows, owners = [], []
        for owner, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            for start in range(0, max(1, len(ids)), max_tokens):
                windows.append(tokenizer.decode(ids[start:start + max_tokens]))
                owners.append(owner)
        if not windows:
            return []
        vectors = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False)
        pooled = []
        for owner in range(len(texts)):
            vector = np.mean([v for v, o in zip(vectors, owners) if o == owner], axis=0)
            norm = np.linalg.norm(vector)
            pooled.append((vector / norm if norm else vector).tolist())
        return pooled
