from __future__ import annotations

import argparse
import hashlib
import os
from typing import Iterable, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_svg_files(root: str) -> Iterable[Tuple[str, str]]:
    for r, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith((".svg", ".svgz")):
                yield os.path.join(r, name), name


def load_sources(path: str) -> List[str]:
    lines: List[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
    return lines


def merge_svgs(inputs: List[str], out_dir: str) -> None:
    ensure_dir(out_dir)
    out_sources = os.path.join(out_dir, "sources.tsv")
    seen_hash: dict[str, str] = {}

    # preload existing outputs to avoid re-copying
    for path, name in iter_svg_files(out_dir):
        try:
            with open(path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            seen_hash[h] = name
        except Exception:
            continue

    with open(out_sources, "a", encoding="utf-8") as map_f:
        for src_root in inputs:
            src_sources = os.path.join(src_root, "sources.tsv")
            provenance = load_sources(src_sources)
            prov_map: dict[str, Tuple[str, str]] = {}
            for ln in provenance:
                parts = ln.split("\t")
                if len(parts) >= 3:
                    prov_map[parts[0]] = (parts[1], parts[2])
            for src_path, src_name in iter_svg_files(src_root):
                try:
                    with open(src_path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue
                h = hashlib.sha256(data).hexdigest()
                if h in seen_hash:
                    continue
                # write new file; preserve original basename when possible
                out_name = src_name
                base, ext = os.path.splitext(out_name)
                idx = 1
                while os.path.exists(os.path.join(out_dir, out_name)):
                    out_name = f"{base}_{idx}{ext}"
                    idx += 1
                out_path = os.path.join(out_dir, out_name)
                try:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    seen_hash[h] = out_name
                    # provenance: if available use original
                    if src_name in prov_map:
                        title, url = prov_map[src_name]
                        map_f.write(f"{out_name}\t{title}\t{url}\n")
                    else:
                        map_f.write(f"{out_name}\tLOCAL\t{src_path}\n")
                    map_f.flush()
                except Exception:
                    continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SVGs from multiple subdirs into one with dedupe")
    p.add_argument("--inputs", nargs="+", required=True, help="Input directories containing SVGs")
    p.add_argument("--out-dir", type=str, default="data/svgs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_svgs(args.inputs, args.out_dir)
    print(f"Merged into {args.out_dir}")


if __name__ == "__main__":
    main()


