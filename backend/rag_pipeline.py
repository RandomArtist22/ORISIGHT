from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .embedding import chunk_text, embed_texts, stable_id


logger = logging.getLogger("orisight.rag")


class MedicalRAG:
    COLLECTION_NAME = "medical_knowledge"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any | None = None
        self.collection: Any | None = None
        self._memory_index: dict[str, dict[str, Any]] = {}
        self._chroma_available = False

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=self.settings.chroma_medical_path)
            self._refresh_collection_handle()
            self._chroma_available = True
        except Exception as exc:
            logger.warning(
                "ChromaDB unavailable, using in-memory RAG fallback: %s",
                exc,
            )

    @property
    def chroma_available(self) -> bool:
        return self._chroma_available

    @property
    def storage_backend(self) -> str:
        return "chroma" if self._chroma_available else "memory"

    def reinitialize(self) -> None:
        if not self._chroma_available or self.collection is None or self.client is None:
            self._memory_index = {}
            return

        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._refresh_collection_handle()

    def index_raw_documents(self, reindex: bool = False) -> dict[str, Any]:
        if reindex:
            self.reinitialize()

        docs = self._load_raw_documents(Path(self.settings.raw_docs_dir))
        chunk_rows: list[dict[str, Any]] = []

        for doc in docs:
            chunks = chunk_text(doc["text"], chunk_size=500, overlap=50)
            for idx, chunk in enumerate(chunks):
                chunk_id = stable_id([doc["id"], str(idx), chunk[:120]])
                chunk_rows.append(
                    {
                        "id": chunk_id,
                        "document": chunk,
                        "metadata": {
                            "source": doc.get("source", "unknown"),
                            "title": doc.get("title", ""),
                            "url": doc.get("url", ""),
                            "chunk_index": idx,
                        },
                    }
                )

        if not self._chroma_available or self.collection is None:
            before = len(self._memory_index)
            for row in chunk_rows:
                self._memory_index.setdefault(row["id"], row)
            added = len(self._memory_index) - before
            return {
                "documents": len(docs),
                "chunks_indexed": len(chunk_rows),
                "chunks_added": max(0, added),
                "storage": "memory",
            }

        if not chunk_rows:
            return {"documents": len(docs), "chunks_indexed": 0, "chunks_added": 0}

        all_ids = [row["id"] for row in chunk_rows]
        existing_ids = set()

        try:
            existing = self._run_with_collection_retry(
                "get",
                lambda: self.collection.get(ids=all_ids) if self.collection is not None else {"ids": []},
            )
            existing_ids = set(existing.get("ids", []))
        except Exception as exc:
            logger.exception("Failed to read existing chunk IDs from Chroma: %s", exc)
            existing_ids = set()

        new_rows = [row for row in chunk_rows if row["id"] not in existing_ids]
        if not new_rows:
            return {
                "documents": len(docs),
                "chunks_indexed": len(chunk_rows),
                "chunks_added": 0,
            }

        texts = [row["document"] for row in new_rows]
        try:
            embeddings = embed_texts(texts, self.settings.sentence_model_name)
        except Exception as exc:
            logger.exception("Failed to generate text embeddings: %s", exc)
            return {
                "documents": len(docs),
                "chunks_indexed": len(chunk_rows),
                "chunks_added": 0,
                "error": "embedding_generation_failed",
            }

        batch_size = 64
        added_count = 0
        for start in range(0, len(new_rows), batch_size):
            stop = start + batch_size
            batch = new_rows[start:stop]
            try:
                self._run_with_collection_retry(
                    "add",
                    lambda: self.collection.add(
                        ids=[row["id"] for row in batch],
                        documents=[row["document"] for row in batch],
                        metadatas=[row["metadata"] for row in batch],
                        embeddings=embeddings[start:stop],
                    )
                    if self.collection is not None
                    else None,
                )
                added_count += len(batch)
            except Exception as exc:
                logger.exception("Failed to add chunk batch to Chroma: %s", exc)
                return {
                    "documents": len(docs),
                    "chunks_indexed": len(chunk_rows),
                    "chunks_added": added_count,
                    "error": "chroma_add_failed",
                }

        return {
            "documents": len(docs),
            "chunks_indexed": len(chunk_rows),
            "chunks_added": added_count,
        }

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if not self._chroma_available or self.collection is None:
            return self._retrieve_from_memory(query, top_k)

        try:
            query_embedding = embed_texts([query], self.settings.sentence_model_name)[0]
        except Exception as exc:
            logger.exception("Failed to embed RAG query, returning empty: %s", exc)
            return []

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if self._is_collection_missing(exc):
                logger.warning("Chroma collection handle stale; refreshing and retrying query")
                self._refresh_collection_handle()
                if self.collection is not None:
                    try:
                        result = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"],
                        )
                    except Exception as retry_exc:
                        logger.exception(
                            "Chroma retry failed, falling back to memory retrieval: %s",
                            retry_exc,
                        )
                        return self._retrieve_from_memory(query, top_k)
                else:
                    return self._retrieve_from_memory(query, top_k)
            else:
                logger.exception("Chroma query failed, falling back to memory retrieval: %s", exc)
                return self._retrieve_from_memory(query, top_k)

        try:
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
        except Exception:
            return []

        out: list[dict[str, Any]] = []
        for doc, meta, distance in zip(docs, metas, distances):
            out.append(
                {
                    "source": meta.get("source", "unknown") if isinstance(meta, dict) else "unknown",
                    "title": meta.get("title", "") if isinstance(meta, dict) else "",
                    "url": meta.get("url", "") if isinstance(meta, dict) else "",
                    "chunk": doc,
                    "score": round(max(0.0, 1.0 - float(distance)), 4),
                }
            )
        return out

    def _refresh_collection_handle(self) -> None:
        if self.client is None:
            return
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _run_with_collection_retry(self, stage: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            if not self._is_collection_missing(exc):
                raise
            logger.warning("Chroma collection handle stale during %s; refreshing and retrying", stage)
            self._refresh_collection_handle()
            return fn()

    @staticmethod
    def _is_collection_missing(exc: Exception) -> bool:
        message = str(exc).lower()
        return "does not exist" in message or "notfounderror" in type(exc).__name__.lower()

    @staticmethod
    def _load_raw_documents(raw_dir: Path) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for path in sorted(raw_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            text = payload.get("text", "").strip()
            if not text:
                continue

            docs.append(
                {
                    "id": str(payload.get("id", path.stem)),
                    "source": payload.get("source", "unknown"),
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "text": text,
                }
            )
        return docs

    def _retrieve_from_memory(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._memory_index:
            return []

        query_tokens = set(query.lower().split())
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._memory_index.values():
            doc = str(row.get("document", ""))
            doc_tokens = set(doc.lower().split())
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(1, len(query_tokens))
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, row in scored[:top_k]:
            meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            out.append(
                {
                    "source": str(meta.get("source", "memory")),
                    "title": str(meta.get("title", "")),
                    "url": str(meta.get("url", "")),
                    "chunk": str(row.get("document", "")),
                    "score": round(float(score), 4),
                }
            )
        return out
