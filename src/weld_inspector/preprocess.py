from __future__ import annotations

import cv2
import numpy as np

from .config import ImagePreprocessSettings


def _normalize_odd_kernel_size(kernel_size: int) -> int:
    normalized = max(1, int(kernel_size))
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def _normalize_tile_grid_size(tile_grid_size: int) -> int:
    return max(1, int(tile_grid_size))


def _apply_unsharp_mask(image: np.ndarray, kernel_size: int, amount: float) -> np.ndarray:
    if amount <= 0:
        return image

    normalized_kernel = _normalize_odd_kernel_size(kernel_size)
    if normalized_kernel <= 1:
        return image

    blurred = cv2.GaussianBlur(image, (normalized_kernel, normalized_kernel), 0)
    sharpened = cv2.addWeighted(image, 1.0 + float(amount), blurred, -float(amount), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_image_preprocess(image: np.ndarray, settings: ImagePreprocessSettings) -> np.ndarray:
    if not settings.is_active:
        return image

    if settings.mode != "grayscale_weld":
        raise ValueError(f"Unsupported preprocess mode: {settings.mode}")

    if image.ndim == 2:
        grayscale = image
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=float(settings.clahe_clip_limit),
        tileGridSize=(
            _normalize_tile_grid_size(settings.clahe_tile_grid_size),
            _normalize_tile_grid_size(settings.clahe_tile_grid_size),
        ),
    )
    enhanced = clahe.apply(grayscale)
    enhanced = _apply_unsharp_mask(
        image=enhanced,
        kernel_size=settings.blur_kernel_size,
        amount=settings.unsharp_amount,
    )
    normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
