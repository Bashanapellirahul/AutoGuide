"""Fixed-token and section-aware chunking with stable IDs and provenance."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import tiktoken
from rag.config import CHUNKS_DIR, PROCESSED_DIR


def token_windows(text, size=500, overlap=100):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    for start in range(0, len(tokens), size - overlap):
        # Decode complete Unicode characters even at a token boundary.
        yield encoding.decode(tokens[start:start + size])
        if start + size >= len(tokens):
            break


def section_texts(document):
    text = document.get("text", "")
    headings = [h.get("text", "") if isinstance(h, dict) else h
                for h in document.get("headings", [])]
    headings = [str(h).strip().rstrip(":") for h in headings if h]
    if not headings:
        return [text]
    pattern = r"(?m)(?=^\s*(?:" + "|".join(re.escape(h) for h in headings) + r"):?\s*$)"
    return [part.strip() for part in re.split(pattern, text) if part.strip()]


def chunk_document(document, strategy="structure", chunk_size=500, overlap=100):
    if strategy not in {"fixed", "structure"}:
        raise ValueError("Unknown chunking strategy.")
    text = document.get("text", "").strip()
    if not text:
        return []
    # Validate even when the document is short.
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    sections = [text] if strategy == "fixed" else section_texts(document)
    encoding = tiktoken.get_encoding("cl100k_base")
    pieces, current = [], ""
    for section in sections:
        combined = f"{current}\n\n{section}".strip()
        if current and len(encoding.encode(combined, disallowed_special=())) > chunk_size:
            pieces.extend(token_windows(current, chunk_size, overlap))
            current = section
        else:
            current = combined
    if current:
        pieces.extend(token_windows(current, chunk_size, overlap))
    doc_id = str(document.get("id") or document.get("doc_id") or hashlib.sha256(text.encode()).hexdigest()[:20])
    metadata = {k: v for k, v in document.items() if k not in {"text", "id", "pages", "tables"}}
    metadata["url"] = document.get("url") or document.get("source_url", "")
    chunks = []
    for index, piece in enumerate(pieces):
        identity = f"{doc_id}|{strategy}|{chunk_size}|{overlap}|{index}|{piece}"
        chunks.append({**metadata, "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                       "document_id": doc_id, "chunk_index": index,
                       "strategy": strategy, "text": piece,
                       "token_count": len(encoding.encode(piece, disallowed_special=()))})
    return chunks


def build_chunks(strategy="structure", chunk_size=500, overlap=100,
                 processed_dir=PROCESSED_DIR, output_dir=CHUNKS_DIR):
    documents = []
    for path in sorted(Path(processed_dir).rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        documents.extend(data if isinstance(data, list) else [data])
    chunks = [chunk for doc in documents
              for chunk in chunk_document(doc, strategy, chunk_size, overlap)]
    if not chunks:
        raise ValueError("No processed documents. Run python ingest.py first.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "chunks.jsonl.tmp"
    temporary.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "chunks.jsonl")
    print(f"Created {len(chunks)} {strategy} chunks from {len(documents)} documents.")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    build_chunks(args.strategy, args.chunk_size, args.overlap)
