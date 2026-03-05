from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=2)
def _load_clip(model_name: str, allow_downloads: bool):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    local_files_only = not allow_downloads
    model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return model, processor


def encode_image(image_path: str | Path, model_name: str, allow_downloads: bool = False) -> list[float]:
    image = Image.open(image_path).convert("RGB")

    try:
        model, processor = _load_clip(model_name, allow_downloads)
        import torch

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].cpu().tolist()
    except Exception:
        return _fallback_hist_embedding(image)


def _fallback_hist_embedding(image: Image.Image) -> list[float]:
    """Fallback embedding when CLIP cannot be loaded.

    Returns a deterministic 512-d vector derived from RGB histograms.
    """
    arr = np.asarray(image.resize((224, 224)), dtype=np.float32)
    channels = [arr[..., i] for i in range(3)]
    features: list[float] = []

    for channel in channels:
        hist, _ = np.histogram(channel, bins=64, range=(0, 255), density=True)
        features.extend(hist.tolist())

    vector = np.array(features, dtype=np.float32)
    if vector.size < 512:
        pad = np.zeros(512 - vector.size, dtype=np.float32)
        vector = np.concatenate([vector, pad])
    elif vector.size > 512:
        vector = vector[:512]

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()
