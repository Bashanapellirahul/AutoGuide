import hashlib
import json
from pathlib import Path

import tiktoken


INPUT_DIR = Path("data/processed/api")
OUTPUT_DIR = Path("data/chunks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """Count tokens in a text."""

    return len(tokenizer.encode(text))


def generate_chunk_id(document_id, strategy, index):
    """Generate a unique chunk identifier."""

    value = f"{document_id}_{strategy}_{index}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:20]


def fixed_size_chunking(document):
    """Split text into fixed-size overlapping chunks."""

    text = document.get("text", "")
    tokens = tokenizer.encode(text)

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for index, start in enumerate(range(0, len(tokens), step)):

        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            continue

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "chunk_id": generate_chunk_id(
                document["id"], "fixed", index
            ),
            "document_id": document["id"],
            "strategy": "fixed",
            "chunk_index": index,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "title": document.get("title", ""),
            "source_url": document.get("url", ""),
            "campaign_number": document.get(
                "campaign_number", ""
            ),
            "vehicle": document.get("vehicle", {})
        })

    return chunks


def structure_aware_chunking(document):
    """Keep recall information together when possible."""

    text = document.get("text", "")

    tokens = tokenizer.encode(text)

    chunks = []

    if len(tokens) <= CHUNK_SIZE:

        chunks.append({
            "chunk_id": generate_chunk_id(
                document["id"], "structure", 0
            ),
            "document_id": document["id"],
            "strategy": "structure",
            "chunk_index": 0,
            "text": text,
            "token_count": len(tokens),
            "title": document.get("title", ""),
            "source_url": document.get("url", ""),
            "campaign_number": document.get(
                "campaign_number", ""
            ),
            "vehicle": document.get("vehicle", {})
        })

    else:

        # Fallback to fixed-size splitting for long documents.
        fixed_chunks = fixed_size_chunking(document)

        for index, chunk in enumerate(fixed_chunks):

            chunk["strategy"] = "structure"

            chunk["chunk_id"] = generate_chunk_id(
                document["id"], "structure", index
            )

            chunks.append(chunk)

    return chunks


def process_documents():

    files = sorted(INPUT_DIR.glob("*.json"))

    if not files:
        print("No normalized documents found.")
        return

    all_chunks = []

    for file_path in files:

        with file_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        fixed_chunks = fixed_size_chunking(document)

        structure_chunks = structure_aware_chunking(document)

        all_chunks.extend(fixed_chunks)
        all_chunks.extend(structure_chunks)

        print(
            f"{file_path.name}: "
            f"{len(fixed_chunks)} fixed chunks, "
            f"{len(structure_chunks)} structure chunks"
        )

    output_file = OUTPUT_DIR / "recall_chunks.json"

    output_file.write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nChunking completed successfully.")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    process_documents()