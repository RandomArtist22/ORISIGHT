from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image


def generate_gradcam_style_heatmap(
    image_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.45,
) -> str:
    """Generate a Grad-CAM style overlay using color + edge saliency heuristics.

    This is a hackathon-friendly explainability proxy, not a clinical segmentation model.
    """
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0

    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue

    grad_y, grad_x = np.gradient(gray)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    edge = _normalize(edge)

    color_saliency = _normalize((1.6 * red - 0.4 * green - 0.2 * blue))
    heat = _normalize(0.6 * color_saliency + 0.4 * edge)

    color_map = cm.get_cmap("jet")(heat)[..., :3]
    overlay = np.clip((1.0 - alpha) * arr + alpha * color_map, 0, 1)

    output = Image.fromarray((overlay * 255).astype(np.uint8))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return str(output_path)


def _normalize(arr: np.ndarray) -> np.ndarray:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)
