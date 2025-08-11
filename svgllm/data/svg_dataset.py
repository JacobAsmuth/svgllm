from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from ..renderer import render_svg_to_rgb, is_svg_valid


@dataclass
class SvgExample:
    image: Image.Image
    svg_text: str
    filename: str


class SvgSftDataset:
    """Dataset that loads SVG files from a directory, renders them to images, and returns pairs.

    This performs simple on-disk caching of renders under `{root}/png{width}x{height}`.
    """

    def __init__(
        self,
        root_dir: str,
        *,
        image_size: Tuple[int, int] = (256, 256),
        max_items: Optional[int] = None,
    ) -> None:
        self.root_dir = root_dir
        self.image_size = image_size
        self.max_items = max_items
        self.svg_paths: List[str] = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".svg", ".svgz"))
        ]
        self.svg_paths.sort()
        if max_items is not None:
            self.svg_paths = self.svg_paths[:max_items]

        w, h = image_size
        self.cache_dir = os.path.join(root_dir, f"png{w}x{h}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.svg_paths)

    def _render_cached(self, svg_text: str, filename: str) -> Image.Image:
        w, h = self.image_size
        name_base = os.path.splitext(os.path.basename(filename))[0]
        cache_path = os.path.join(self.cache_dir, f"{name_base}.png")
        if os.path.exists(cache_path):
            return Image.open(cache_path).convert("RGB")
        rgb = render_svg_to_rgb(svg_text, size=(w, h))
        img = Image.fromarray(rgb, mode="RGB")
        img.save(cache_path)
        return img

    def __getitem__(self, idx: int) -> SvgExample:  # type: ignore[override]
        path = self.svg_paths[idx]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            svg_text = f.read()
        if not is_svg_valid(svg_text):
            # Replace invalid with a tiny blank SVG to keep batch shape
            svg_text = (
                "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'></svg>"
            )
        img = self._render_cached(svg_text, path)
        return SvgExample(image=img, svg_text=svg_text, filename=os.path.basename(path))


