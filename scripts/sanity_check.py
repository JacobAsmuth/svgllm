from __future__ import annotations

import argparse
import os
from glob import glob
from typing import List, Tuple

from PIL import Image
import numpy as np

from svgllm.renderer import render_svg_to_rgb, is_svg_valid


def find_svg_files(root_dir: str, recursive: bool = True) -> List[str]:
    if recursive:
        paths = glob(os.path.join(root_dir, "*.svg"))
    else:
        paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".svg"))
        ]
    paths = sorted(paths)
    return paths


def tile_images(images: List[Image.Image], tile_size: Tuple[int, int]) -> Image.Image:
    if not images:
        return Image.new("RGB", tile_size, (255, 255, 255))
    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
    w, h = tile_size
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for idx, img in enumerate(images):
        col = idx % cols
        row = idx // cols
        canvas.paste(img, (col * w, row * h))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check SVG dataset and render a preview grid")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", type=str, default="/content/svg_preview.png")
    ap.add_argument("--recursive", action="store_true", default=True)
    args = ap.parse_args()

    paths = find_svg_files(args.data_dir, recursive=args.recursive)
    print(f"Found {len(paths)} SVG files under {args.data_dir}")
    for p in paths[: min(10, len(paths))]:
        print(" -", os.path.relpath(p, args.data_dir))

    renders: List[Image.Image] = []
    for p in paths[: args.limit]:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                svg_text = f.read()
            if not is_svg_valid(svg_text):
                continue
            rgb = render_svg_to_rgb(svg_text, size=(args.size, args.size))
            renders.append(Image.fromarray(rgb, mode="RGB"))
        except Exception as e:
            print(f"[warn] failed to render {p}: {e}")

    if not renders:
        print("No previews rendered.")
        return

    grid = tile_images(renders, (args.size, args.size))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    grid.save(args.out)
    print(f"Wrote preview grid: {args.out}")


if __name__ == "__main__":
    main()


