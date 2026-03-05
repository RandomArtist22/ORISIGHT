from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..embedding import stable_id
from .clip_encoder import encode_image


logger = logging.getLogger("orisight.similarity")


class LesionSimilarityService:
    COLLECTION_NAME = "lesion_cases"

    def __init__(self, chroma_path: str, clip_model_name: str, allow_model_downloads: bool = False) -> None:
        self.clip_model_name = clip_model_name
        self.allow_model_downloads = allow_model_downloads
        self.client: Any | None = None
        self.collection: Any | None = None
        self._memory_cases: dict[str, dict[str, Any]] = {}
        self._chroma_available = False

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=chroma_path)
            self._refresh_collection_handle()
            self._chroma_available = True
        except Exception as exc:
            logger.warning(
                "ChromaDB unavailable for lesion similarity, using in-memory fallback: %s",
                exc,
            )

    def upsert_case(
        self,
        case_id: str,
        image_embedding: list[float],
        diagnosis: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = {"diagnosis": diagnosis, **(metadata or {})}
        if self._chroma_available and self.collection is not None:
            try:
                self._run_with_collection_retry(
                    "upsert",
                    lambda: self.collection.upsert(
                        ids=[case_id],
                        embeddings=[image_embedding],
                        metadatas=[meta],
                        documents=[diagnosis],
                    )
                    if self.collection is not None
                    else None,
                )
                return
            except Exception as exc:
                logger.exception(
                    "Failed to upsert case into Chroma similarity index; using memory fallback: %s",
                    exc,
                )

        self._memory_cases[case_id] = {
            "embedding": image_embedding,
            "diagnosis": diagnosis,
            "metadata": meta,
        }

    def upsert_from_image(
        self,
        image_path: str | Path,
        diagnosis: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        embedding = encode_image(
            image_path,
            self.clip_model_name,
            allow_downloads=self.allow_model_downloads,
        )
        case_id = stable_id([str(image_path), diagnosis])
        self.upsert_case(case_id, embedding, diagnosis, metadata)
        return case_id

    def query_similar(
        self,
        image_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not self._chroma_available or self.collection is None:
            return self._query_memory(image_embedding, top_k)

        try:
            result = self._run_with_collection_retry(
                "query",
                lambda: self.collection.query(
                    query_embeddings=[image_embedding],
                    n_results=top_k,
                    include=["metadatas", "distances", "documents"],
                )
                if self.collection is not None
                else {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]},
            )
        except Exception as exc:
            logger.exception("Chroma similarity query failed, using memory fallback: %s", exc)
            return self._query_memory(image_embedding, top_k)

        ids = result.get("ids", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        docs = result.get("documents", [[]])[0]

        rows: list[dict[str, Any]] = []
        for case_id, meta, distance, doc in zip(ids, metas, distances, docs):
            diagnosis = "Unknown"
            metadata = {}
            if isinstance(meta, dict):
                diagnosis = str(meta.get("diagnosis", doc or "Unknown"))
                metadata = meta

            similarity = max(0.0, 1.0 - float(distance))
            rows.append(
                {
                    "case_id": case_id,
                    "diagnosis": diagnosis,
                    "similarity": round(similarity, 4),
                    "metadata": metadata,
                }
            )

        return rows

    def ingest_seed_directory(self, seed_dir: str | Path) -> dict[str, Any]:
        seed_path = Path(seed_dir)
        if not seed_path.exists():
            return {"seed_dir": str(seed_path), "ingested": 0}

        diagnosis_map = _load_metadata_map(seed_path / "metadata.json")
        ingested = 0

        for file in seed_path.rglob("*"):
            if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            diagnosis = diagnosis_map.get(file.name, _infer_diagnosis_from_filename(file.name))
            if not diagnosis:
                diagnosis = "Unknown lesion"
            self.upsert_from_image(file, diagnosis, {"source": "seed", "filename": file.name})
            ingested += 1

        return {"seed_dir": str(seed_path), "ingested": ingested}

    def _query_memory(self, image_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not self._memory_cases:
            return []

        query = np.asarray(image_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for case_id, payload in self._memory_cases.items():
            candidate = np.asarray(payload.get("embedding", []), dtype=np.float32)
            if candidate.size == 0:
                continue
            if candidate.shape != query.shape:
                continue
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm == 0:
                continue
            cosine = float(np.dot(query, candidate) / (query_norm * candidate_norm))
            scored.append((cosine, case_id, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[dict[str, Any]] = []
        for cosine, case_id, payload in scored[:top_k]:
            diagnosis = str(payload.get("diagnosis", "Unknown"))
            metadata = payload.get("metadata", {})
            out.append(
                {
                    "case_id": case_id,
                    "diagnosis": diagnosis,
                    "similarity": round(max(0.0, cosine), 4),
                    "metadata": metadata if isinstance(metadata, dict) else {},
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
            logger.warning("Similarity collection handle stale during %s; refreshing and retrying", stage)
            self._refresh_collection_handle()
            return fn()

    @staticmethod
    def _is_collection_missing(exc: Exception) -> bool:
        message = str(exc).lower()
        return "does not exist" in message or "notfounderror" in type(exc).__name__.lower()


def _load_metadata_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in payload.items():
        out[str(key)] = str(value)
    return out


def _infer_diagnosis_from_filename(filename: str) -> str:
    lowered = filename.lower()
    if "osmf" in lowered or "submucous" in lowered:
        return "Oral Submucous Fibrosis"
    if "leuk" in lowered or "white" in lowered:
        return "Leukoplakia"
    if "eryth" in lowered or "red" in lowered:
        return "Erythroplakia"
    if "lichen" in lowered:
        return "Oral Lichen Planus"
    if "scc" in lowered or "carcinoma" in lowered or "cancer" in lowered:
        return "Oral Squamous Cell Carcinoma"
    return ""
