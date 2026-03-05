from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=2)
def _load_blip(
    model_name: str,
    allow_downloads: bool,
):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    local_files_only = not allow_downloads
    processor = BlipProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    return processor, model


def generate_lesion_caption(
    image_path: str | Path,
    model_name: str,
    allow_downloads: bool = False,
) -> str:
    image = Image.open(image_path).convert("RGB")

    try:
        processor, model = _load_blip(model_name, allow_downloads)
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(output[0], skip_special_tokens=True)
        caption = " ".join(caption.split())
        if caption:
            return caption
    except Exception:
        pass

    return _fallback_caption(image)


def _fallback_caption(image: Image.Image) -> str:
    arr = np.asarray(image.resize((256, 256)), dtype=np.float32) / 255.0
    redness = float(arr[..., 0].mean() - 0.5 * (arr[..., 1].mean() + arr[..., 2].mean()))
    brightness = float(arr.mean())

    tone = "reddish" if redness > 0.02 else "pale"
    lighting = "well lit" if brightness > 0.5 else "dim"
    return (
        f"Oral cavity photo with {tone} mucosal region, {lighting}, "
        "possible patch-like lesion visible."
    )
