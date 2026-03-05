from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Iterable

import numpy as np

logger = logging.getLogger("orisight.embedding")
_LAST_ENCODER_ERROR: str | None = None


@lru_cache(maxsize=2)
def load_text_encoder(model_name: str):
    global _LAST_ENCODER_ERROR
    try:
        from sentence_transformers import SentenceTransformer

        _LAST_ENCODER_ERROR = None
        return SentenceTransformer(model_name)
    except Exception as exc:
        _LAST_ENCODER_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning("Failed to load text encoder '%s': %s", model_name, _LAST_ENCODER_ERROR)
        return None


def get_last_text_encoder_error() -> str | None:
    return _LAST_ENCODER_ERROR


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    model = load_text_encoder(model_name)
    if model is None:
        return [_fallback_text_embedding(text) for text in texts]

    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = words[start:end]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if end >= len(words):
            break

    return chunks


def stable_id(parts: Iterable[str]) -> str:
    data = "::".join(parts)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _fallback_text_embedding(text: str, dim: int = 384) -> list[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vec.tolist()

    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()
