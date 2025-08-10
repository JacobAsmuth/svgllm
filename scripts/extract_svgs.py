from __future__ import annotations

import argparse
import io
import os
import tarfile
import zipfile
from typing import Iterable, Optional, Tuple

from tqdm import tqdm


SVG_EXTS = (".svg", ".svgz")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name)
    return base.replace("\x00", "_")


def write_file(out_dir: str, filename: str, content: bytes, sources_f) -> Optional[str]:
    fname = sanitize_filename(filename)
    out_path = os.path.join(out_dir, fname)
    # Deduplicate by filename; if exists, append a numeric suffix
    if os.path.exists(out_path):
        root, ext = os.path.splitext(fname)
        idx = 1
        while os.path.exists(os.path.join(out_dir, f"{root}_{idx}{ext}")):
            idx += 1
        fname = f"{root}_{idx}{ext}"
        out_path = os.path.join(out_dir, fname)
    try:
        with open(out_path, "wb") as f:
            f.write(content)
        sources_f.write(f"{fname}\tLOCAL\tN/A\n")
        sources_f.flush()
        return out_path
    except Exception:
        return None


def extract_from_zip(zip_path: str, out_dir: str, max_files: int) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf, open(os.path.join(out_dir, "sources.tsv"), "a", encoding="utf-8") as sources_f:
        for info in tqdm(zf.infolist(), desc="zip entries"):
            if count >= max_files:
                break
            name = info.filename
            if not name.lower().endswith(SVG_EXTS):
                continue
            with zf.open(info, "r") as f:
                content = f.read()
            if write_file(out_dir, name, content, sources_f) is not None:
                count += 1
    return count


def extract_from_tar(tar_path: str, out_dir: str, max_files: int) -> int:
    count = 0
    mode = "r:*"  # auto-detect compression
    with tarfile.open(tar_path, mode) as tf, open(os.path.join(out_dir, "sources.tsv"), "a", encoding="utf-8") as sources_f:
        members = tf.getmembers()
        for m in tqdm(members, desc="tar entries"):
            if count >= max_files:
                break
            if not m.isfile():
                continue
            name = m.name
            if not name.lower().endswith(SVG_EXTS):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            content = f.read()
            if write_file(out_dir, name, content, sources_f) is not None:
                count += 1
    return count


def extract_from_dir(in_dir: str, out_dir: str, max_files: int) -> int:
    count = 0
    with open(os.path.join(out_dir, "sources.tsv"), "a", encoding="utf-8") as sources_f:
        for root, _, files in os.walk(in_dir):
            for name in files:
                if count >= max_files:
                    return count
                if not name.lower().endswith(SVG_EXTS):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                    if write_file(out_dir, name, content, sources_f) is not None:
                        count += 1
                except Exception:
                    continue
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract .svg/.svgz files from an archive or directory")
    p.add_argument("input", type=str, help="Path to .zip/.tar(.gz|.xz) archive or a directory")
    p.add_argument("--out-dir", type=str, default="data/extracted_svgs", help="Output directory")
    p.add_argument("--max-files", type=int, default=1000000, help="Max number of files to extract")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    path = args.input
    if os.path.isdir(path):
        n = extract_from_dir(path, args.out_dir, args.max_files)
    elif zipfile.is_zipfile(path):
        n = extract_from_zip(path, args.out_dir, args.max_files)
    elif tarfile.is_tarfile(path):
        n = extract_from_tar(path, args.out_dir, args.max_files)
    else:
        raise SystemExit("Unsupported input: must be a directory, .zip, or tar archive")
    print(f"Extracted {n} SVG files to {args.out_dir}")


if __name__ == "__main__":
    main()



