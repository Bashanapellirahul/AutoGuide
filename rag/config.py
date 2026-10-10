"""Project paths are independent of the terminal's working directory."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "nhtsa_chunks")
