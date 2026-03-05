from __future__ import annotations

import argparse
import json

try:
    from .config import get_settings
    from .rag_pipeline import MedicalRAG
except ImportError:  # pragma: no cover
    from backend.config import get_settings
    from backend.rag_pipeline import MedicalRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/refresh medical embeddings in ChromaDB")
    parser.add_argument("--reindex", action="store_true", help="Drop and rebuild the medical docs collection")
    args = parser.parse_args()

    settings = get_settings()
    rag = MedicalRAG(settings)
    stats = rag.index_raw_documents(reindex=args.reindex)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
