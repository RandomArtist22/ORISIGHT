#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import get_settings
from backend.embedding import chunk_text, get_last_text_encoder_error, load_text_encoder
from backend.rag_pipeline import MedicalRAG
from backend.scraper import clean_text, run_scraping_pipeline


logger = logging.getLogger("orisight.rag.builder")


def normalize_raw_documents(raw_dir: Path) -> dict[str, int]:
    processed = 0
    updated = 0

    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON: %s", path)
            continue

        text = str(payload.get("text", ""))
        normalized = clean_text(text)
        processed += 1

        if normalized != text:
            payload["text"] = normalized
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            updated += 1

    return {"processed": processed, "updated": updated}


def estimate_chunking(raw_dir: Path, chunk_size: int = 500, overlap: int = 50) -> dict[str, int]:
    docs = 0
    total_chunks = 0

    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        text = str(payload.get("text", "")).strip()
        if not text:
            continue

        docs += 1
        total_chunks += len(chunk_text(text, chunk_size=chunk_size, overlap=overlap))

    return {"documents": docs, "estimated_chunks": total_chunks}


def _install_ml_dependencies(requirements_path: Path) -> None:
    if not requirements_path.exists():
        raise RuntimeError(f"Missing ML requirements file: {requirements_path}")

    logger.info("Installing ML dependencies from %s", requirements_path)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


def ensure_sentence_transformer(
    model_name: str,
    allow_fallback: bool,
    auto_install_ml_deps: bool,
) -> str:
    model = load_text_encoder(model_name)
    if model is None:
        if auto_install_ml_deps:
            requirements_path = ROOT_DIR / "backend" / "requirements-ml.txt"
            try:
                _install_ml_dependencies(requirements_path)
                # Retry after dependency install; the first failed result may be cached.
                try:
                    load_text_encoder.cache_clear()
                except Exception:
                    pass
                model = load_text_encoder(model_name)
            except Exception as exc:
                logger.warning("Auto-install of ML dependencies failed: %s", exc)

    if model is None:
        last_error = get_last_text_encoder_error()
        if last_error:
            logger.warning("Final text encoder load failure: %s", last_error)
        if allow_fallback:
            logger.warning(
                "sentence-transformers unavailable; using fallback embeddings. "
                "Install backend/requirements-ml.txt for model-based embeddings."
            )
            return "fallback"
        raise RuntimeError(
            "sentence-transformers model could not be loaded. "
            "Install optional deps with: pip install -r backend/requirements-ml.txt"
        )

    logger.info("Loaded sentence-transformers model: %s", model_name)
    return "sentence-transformers"


def build_rag_database(
    *,
    pubmed_query: str,
    pubmed_max_results: int,
    reindex: bool,
    model_name: str,
    allow_memory_fallback: bool,
    allow_embedding_fallback: bool,
    auto_install_ml_deps: bool,
) -> dict[str, object]:
    settings = get_settings()
    settings.sentence_model_name = model_name

    start = time.time()

    logger.info("Step 1/5: Scraping sources")
    scrape_stats = run_scraping_pipeline(settings, pubmed_query, pubmed_max_results)
    logger.info(
        "Scraped %s docs, saved %s docs from sources=%s",
        scrape_stats.get("scraped_count", 0),
        scrape_stats.get("saved_count", 0),
        scrape_stats.get("sources", []),
    )

    logger.info("Step 2/5: Cleaning and normalizing raw documents")
    normalize_stats = normalize_raw_documents(Path(settings.raw_docs_dir))
    logger.info(
        "Normalized %s docs (%s updated)",
        normalize_stats["processed"],
        normalize_stats["updated"],
    )

    logger.info("Step 3/5: Chunking documents (chunk_size=500, overlap=50)")
    chunk_stats = estimate_chunking(Path(settings.raw_docs_dir), chunk_size=500, overlap=50)
    logger.info(
        "Prepared chunk plan: %s docs -> ~%s chunks",
        chunk_stats["documents"],
        chunk_stats["estimated_chunks"],
    )

    logger.info("Step 4/5: Ensuring embedding model (%s)", model_name)
    embedding_backend = ensure_sentence_transformer(
        model_name=model_name,
        allow_fallback=allow_embedding_fallback,
        auto_install_ml_deps=auto_install_ml_deps,
    )

    logger.info("Step 5/5: Embedding + storing in ChromaDB")
    rag = MedicalRAG(settings)
    if not rag.chroma_available and not allow_memory_fallback:
        raise RuntimeError(
            "ChromaDB is unavailable in this environment. "
            "Use Python 3.11/3.12 or rerun with --allow-memory-fallback."
        )

    ingest_stats = rag.index_raw_documents(reindex=reindex)
    elapsed = round(time.time() - start, 2)

    summary: dict[str, object] = {
        "elapsed_seconds": elapsed,
        "storage_backend": rag.storage_backend,
        "embedding_backend": embedding_backend,
        "scrape": scrape_stats,
        "normalize": normalize_stats,
        "chunking": chunk_stats,
        "ingest": ingest_stats,
        "model": model_name,
    }

    logger.info(
        "RAG build complete in %ss | storage=%s | embedding=%s | docs=%s | chunks_added=%s",
        elapsed,
        rag.storage_backend,
        embedding_backend,
        chunk_stats["documents"],
        ingest_stats.get("chunks_added", 0),
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ORISIGHT RAG database: scrape -> clean -> chunk -> embed -> store",
    )
    parser.add_argument(
        "--pubmed-query",
        default="oral potentially malignant disorders",
        help="PubMed query term",
    )
    parser.add_argument(
        "--pubmed-max-results",
        type=int,
        default=8,
        help="Number of PubMed abstracts to fetch",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers embedding model",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Drop and rebuild the existing medical knowledge collection",
    )
    parser.add_argument(
        "--allow-memory-fallback",
        action="store_true",
        help="Allow in-memory storage if ChromaDB cannot initialize",
    )
    parser.add_argument(
        "--allow-embedding-fallback",
        action="store_true",
        help="Allow fallback hashed embeddings if sentence-transformers is unavailable",
    )
    parser.add_argument(
        "--auto-install-ml-deps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-install backend/requirements-ml.txt when sentence-transformers is missing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    summary = build_rag_database(
        pubmed_query=args.pubmed_query,
        pubmed_max_results=args.pubmed_max_results,
        reindex=args.reindex,
        model_name=args.model,
        allow_memory_fallback=args.allow_memory_fallback,
        allow_embedding_fallback=args.allow_embedding_fallback,
        auto_install_ml_deps=args.auto_install_ml_deps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
