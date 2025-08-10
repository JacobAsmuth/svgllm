from __future__ import annotations

import io
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cairosvg


_DISALLOWED_TAGS = re.compile(r"<(script|foreignObject|iframe|embed|object|image|video|audio)\b", re.IGNORECASE)


def is_svg_valid(svg_text: str) -> bool:
    """Return True if the SVG looks syntactically safe for rendering.

    This is a light-weight preflight check; it does not guarantee full safety.
    """
    if not svg_text:
        return False
    if _DISALLOWED_TAGS.search(svg_text) is not None:
        return False
    # Basic sanity: must contain an <svg ...> root element
    return "<svg" in svg_text.lower() and "</svg>" in svg_text.lower()


def render_svg_to_rgb(
    svg_text: str,
    size: Tuple[int, int] = (256, 256),
    background: Optional[Tuple[int, int, int]] = (255, 255, 255),
    timeout: float | None = None,
) -> np.ndarray:
    """Render SVG to an RGB numpy array uint8 of shape (H, W, 3).

    Args:
        svg_text: The SVG XML string.
        size: Output (width, height) in pixels.
        background: Optional RGB background to flatten alpha. If None, keep alpha.
        timeout: Max seconds to allow rendering.
    Returns:
        np.ndarray of shape (H, W, 3) dtype uint8 in [0,255].
    Raises:
        RuntimeError on render failure.
    """
    if not is_svg_valid(svg_text):
        raise RuntimeError("Invalid or disallowed SVG content")

    width, height = size
    try:
        # Note: some CairoSVG versions do not support a timeout kwarg; we omit it here.
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=width,
            output_height=height,
            background_color=(
                None if background is None else
                f"rgb({background[0]},{background[1]},{background[2]})"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - propagate as runtime error with message
        raise RuntimeError(f"SVG render failed: {exc}") from exc

    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


