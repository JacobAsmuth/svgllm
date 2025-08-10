from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from typing import Dict, List, Optional, Set
import subprocess

import requests
from tqdm import tqdm


COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_existing(out_dir: str) -> Set[str]:
    existing: Set[str] = set()
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.lower().endswith((".svg", ".svgz")):
                existing.add(name)
    sources_path = os.path.join(out_dir, "sources.tsv")
    if os.path.exists(sources_path):
        with open(sources_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    existing.add(parts[0])
    return existing


def _sleep(base: float, jitter: float) -> None:
    if base <= 0:
        return
    time.sleep(max(0.0, base * (1.0 + random.uniform(-jitter, jitter))))


def _get(url: str, params: Dict[str, str], headers: Dict[str, str], retries: int) -> requests.Response:
    last: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(10.0, 1.0 * (2 ** i)))
    assert last is not None
    raise last


def fetch_allimages(
    out_dir: str,
    *,
    max_files: int = 200,
    base_delay_s: float = 0.8,
    jitter_frac: float = 0.2,
    max_retries: int = 3,
    user_agent: str = "svgllm-bot/0.1 (contact: jacobasmuth@gmail.com)",
    concurrency: int = 4,
    rclone_remote: Optional[str] = None,
    rclone_dir: Optional[str] = None,
    rclone_batch: int = 10,
) -> None:
    ensure_dir(out_dir)
    t_start = time.perf_counter()
    sources_path = os.path.join(out_dir, "sources.tsv")
    existing = load_existing(out_dir)
    headers = {"User-Agent": user_agent}

    # Use generator=allpages on namespace=6 (File:) to avoid MIME search limits.
    base_params: Dict[str, str] = {
        "action": "query",
        "format": "json",
        "generator": "allpages",
        "gapnamespace": "6",
        "gaplimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|mime|size|timestamp",
        "origin": "*",
    }

    # Producer-consumer: stream listing into a download queue
    q: "queue.Queue[Optional[Dict[str, str]]]" = queue.Queue(maxsize=max(32, concurrency * 8))
    lock = threading.Lock()
    saved_count = 0
    bytes_total = 0

    def worker() -> None:
        nonlocal saved_count, bytes_total
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            it = item
            _sleep(base_delay_s, jitter_frac)
            try:
                t_d0 = time.perf_counter()
                r = _get(it["url"], {}, headers, max_retries)
                content = r.content
                path = os.path.join(out_dir, it["filename"])
                with open(path, "wb") as f:
                    f.write(content)
                secs = time.perf_counter() - t_d0
                row = it["filename"] + "\t" + it["title"] + "\t" + it["url"]
                with lock:
                    if it["filename"] not in existing:
                        map_f.write(row + "\n")
                        map_f.flush()
                        existing.add(it["filename"])
                        pbar.update(1)
                        saved_count += 1
                        bytes_total += len(content)
                        rate_kib = (len(content) / 1024.0) / max(1e-6, secs)
                        print(f"[ok] saved {it['filename']} in {secs*1000:.0f} ms, {len(content)/1024.0:.1f} KiB @ {rate_kib:.1f} KiB/s")
            except Exception:
                print(f"[warn] download failed: {it['url']}")
            finally:
                q.task_done()

    with tqdm(total=max_files, desc="API SVGs", unit="file") as pbar, open(sources_path, "a", encoding="utf-8") as map_f:
        # Start workers
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()

        list_pages = 0
        prefixes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, start_char in enumerate(prefixes):
            if saved_count >= max_files:
                break
            start = f"File:{start_char}"
            end = f"File:{prefixes[i+1]}" if i + 1 < len(prefixes) else None
            cont: Dict[str, str] = {}
            print(f"[info] scanning titles from '{start}' to '{end or 'END'}'")
            while saved_count + q.qsize() < max_files:
                _sleep(base_delay_s, jitter_frac)
                params = dict(base_params)
                params["gapfrom"] = start
                if end is not None:
                    params["gapto"] = end
                merged = {**params, **cont}
                try:
                    t_list0 = time.perf_counter()
                    resp = _get(COMMONS_API, merged, headers, max_retries)
                    data = resp.json()
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] listing failed: {e}")
                    break
                if "error" in data:
                    print(f"[warn] API error: {data['error']}")
                pages = (data.get("query") or {}).get("pages") or {}
                emits = 0
                for page in pages.values():
                    title = page.get("title", "")
                    if not title.lower().endswith((".svg", ".svgz")):
                        continue
                    infos = page.get("imageinfo") or []
                    if not infos:
                        continue
                    info = infos[0]
                    url = info.get("url")
                    if not url:
                        continue
                    filename = os.path.basename(url)
                    if filename in existing:
                        continue
                    q.put({"url": url, "title": title, "filename": filename})
                    emits += 1
                    if saved_count + q.qsize() >= max_files:
                        break
                list_pages += 1
                dt_ms = (time.perf_counter() - t_list0) * 1000.0
                print(f"[timing] list#{list_pages} [{start_char}]: emitted {emits} tasks in {dt_ms:.0f} ms; in-queue {q.qsize()}")
                cont = data.get("continue") or {}
                if not cont:
                    break

        # Signal workers to stop
        for _ in range(concurrency):
            q.put(None)
        q.join()

        # Optional: sync to Google Drive via rclone
        if rclone_remote and rclone_dir:
            remote_path = f"{rclone_remote}:{rclone_dir}"
            try:
                print(f"[sync] rclone copy {out_dir} -> {remote_path}")
                subprocess.run([
                    "rclone", "copy", out_dir, remote_path,
                    "--ignore-existing", "--transfers", str(max(4, concurrency)),
                ], check=False)
            except FileNotFoundError:
                print("[sync] rclone not found; install with: sudo apt-get install -y rclone")

        elapsed = time.perf_counter() - t_start
        if saved_count:
            print(f"[timing] total: {saved_count} files, {bytes_total/1024.0/1024.0:.2f} MiB in {elapsed:.1f}s, {saved_count/elapsed:.2f} files/s")
        else:
            print(f"[timing] no files saved, elapsed {elapsed:.1f}s")


def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download SVGs from Commons via allimages API")
    p.add_argument("--out-dir", type=str, default="data/commons_svgs", help="Output directory")
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--delay-jitter-frac", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--user-agent", type=str, default="svgllm-bot/0.1 (contact: jacobasmuth@gmail.com)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rclone-remote", type=str, default=None, help="Optional rclone remote name (e.g., gdrive)")
    p.add_argument("--rclone-dir", type=str, default=None, help="Path on the remote (e.g., svgllm/commons_svgs_api)")
    p.add_argument("--rclone-batch", type=int, default=10, help="Batch size to trigger rclone sync (not used in final sync)")
    return p


def main() -> None:
    pa = parse_args()
    a = pa.parse_args()
    fetch_allimages(
        out_dir=a.out_dir,
        max_files=a.max_files,
        base_delay_s=a.delay,
        jitter_frac=a.delay_jitter_frac,
        max_retries=a.max_retries,
        user_agent=a.user_agent,
        concurrency=a.concurrency,
        rclone_remote=a.rclone_remote,
        rclone_dir=a.rclone_dir,
        rclone_batch=a.rclone_batch,
    )


if __name__ == "__main__":
    main()


