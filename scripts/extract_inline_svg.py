from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import tempfile
from typing import Iterable, Optional, Tuple

import py7zr
from bs4 import BeautifulSoup

SVG_XMLNS = "http://www.w3.org/2000/svg"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_standalone_svg(svg_fragment: str) -> str:
    # Parse fragment and ensure root has xmlns and sensible size attributes
    soup = BeautifulSoup(svg_fragment, "html.parser")
    svg = soup.find("svg")
    if svg is None:
        return ""
    if not svg.has_attr("xmlns"):
        svg["xmlns"] = SVG_XMLNS
    # If width/height missing but viewBox present, set width/height from it
    if svg.has_attr("viewBox") and (not svg.has_attr("width") or not svg.has_attr("height")):
        parts = str(svg["viewBox"]).replace(",", " ").split()
        if len(parts) == 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                if not svg.has_attr("width"):
                    svg["width"] = str(int(w))
                if not svg.has_attr("height"):
                    svg["height"] = str(int(h))
            except Exception:
                pass
    # Drop disallowed tags for safety
    for tag in svg.find_all(["script", "foreignObject", "iframe", "embed", "object", "video", "audio"]):
        tag.decompose()
    # Serialize outer HTML
    return str(svg)


def iter_html_from_tar(tar_path: str):
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".html"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                html = f.read().decode("utf-8", errors="ignore")
            finally:
                f.close()
            yield m.name, html


def iter_html_from_7z(archive_path: str):
    with py7zr.SevenZipFile(archive_path, "r") as z:
        # Extract nested tar to temp dir, then iterate
        names = [n.filename for n in z.list()]
        tar_name = next((n for n in names if n.endswith(".tar")), None)
        if tar_name is None:
            return
        tmpdir = tempfile.mkdtemp(prefix="inline_svg_")
        z.extract(targets=[tar_name], path=tmpdir)
        tar_path = os.path.join(tmpdir, tar_name)
        yield from iter_html_from_tar(tar_path)


def iter_html_from_dir(root_dir: str):
    for r, _, files in os.walk(root_dir):
        for name in files:
            if not name.endswith(".html"):
                continue
            path = os.path.join(r, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    yield os.path.relpath(path, root_dir), f.read()
            except Exception:
                continue


def extract_inline_svgs(input_path: str, out_dir: str, max_svgs: int) -> int:
    ensure_dir(out_dir)
    map_path = os.path.join(out_dir, "sources.tsv")
    seen_hashes: set[str] = set()
    count = 0

    # Choose iterator based on input type
    if os.path.isdir(input_path):
        html_iter = iter_html_from_dir(input_path)
    elif input_path.endswith(".7z"):
        html_iter = iter_html_from_7z(input_path)
    elif input_path.endswith(".tar") or input_path.endswith(".tar.gz") or input_path.endswith(".tar.xz"):
        html_iter = iter_html_from_tar(input_path)
    else:
        raise SystemExit("Unsupported input: directory, .7z, or .tar(.gz|.xz) expected")

    with open(map_path, "a", encoding="utf-8") as map_f:
        for html_name, html in html_iter:
            if count >= max_svgs:
                break
            # Fast check
            if "<svg" not in html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            svgs = soup.find_all("svg")
            for idx, tag in enumerate(svgs):
                if count >= max_svgs:
                    break
                frag = str(tag)
                standalone = make_standalone_svg(frag)
                if not standalone:
                    continue
                # Deduplicate by hash
                h = hashlib.sha256(standalone.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                # Build filename
                base = os.path.basename(html_name).rsplit(".", 1)[0]
                fname = f"{base}_svg{idx}_{h[:10]}.svg"
                out_path = os.path.join(out_dir, fname)
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(standalone)
                    map_f.write(f"{fname}\tINLINE\t{html_name}\n")
                    map_f.flush()
                    count += 1
                except Exception:
                    continue
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract inline <svg> tags from HTML (dir/.tar/.7z)")
    p.add_argument("input", type=str, help="Directory of HTML files, or a .tar(.gz|.xz), or a .7z containing a tar of HTML")
    p.add_argument("--out-dir", type=str, default="data/inline_svgs", help="Output directory for extracted SVGs")
    p.add_argument("--max-svgs", type=int, default=100000, help="Max number of inline SVGs to extract")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = extract_inline_svgs(args.input, args.out_dir, args.max_svgs)
    print(f"Extracted {n} inline SVGs to {args.out_dir}")


if __name__ == "__main__":
    main()


