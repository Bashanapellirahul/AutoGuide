"""Collect recall/feature sources, normalize, chunk, and optionally index."""
import argparse
from ingestion.api_collector import NHTSAAPICollector
from ingestion.api_normalizer import process_api_documents
from ingestion.feature_collector import collect_features
from rag.chunking import build_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing processed documents")
    parser.add_argument("--features", action="store_true", help="Also collect NHTSA feature publications")
    parser.add_argument("--index", action="store_true", help="Download models and build the semantic index")
    parser.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    args = parser.parse_args()
    if not args.skip_fetch:
        NHTSAAPICollector().collect()
        process_api_documents()
        if args.features:
            print(collect_features())
    chunks = build_chunks(args.strategy)
    if args.index:
        from rag.vector_store import VectorStore
        store = VectorStore(chunks)
        print(f"Indexed {store.collection.count()} chunks.")


if __name__ == "__main__":
    main()
