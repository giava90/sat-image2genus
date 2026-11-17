"""Download files from the TreeSatAI Zenodo record with a hard download cap.

Every byte received is appended to data/download_ledger.csv. A file is only
started if (bytes already downloaded + bytes still needed) stays under the cap.
Partial files are resumed with HTTP Range requests, and the md5 from Zenodo is
checked after each file.

Usage: python src/download.py s2.zip aerial_60m_picea_abies.zip ...
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEDGER = ROOT / "data" / "download_ledger.csv"
RECORD_JSON = ROOT / "data" / "zenodo_meta" / "record_6780578.json"
RECORD_ID = "6780578"
CAP_BYTES = 8_000_000_000  # 8 GB, decimal, as in BRIEF.md


def ledger_total() -> int:
    if not LEDGER.exists():
        return 0
    with LEDGER.open() as fh:
        return sum(int(r["bytes"]) for r in csv.DictReader(fh))


def ledger_add(name: str, nbytes: int, note: str) -> None:
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "file", "bytes", "note"])
        w.writerow([dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), name, nbytes, note])


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


LEVEL1_FILES = [
    "labels.zip", "train_filenames.lst", "test_filenames.lst", "220629_doc_TreeSatAI_benchmark_archive.pdf",
    "s2.zip", "aerial_60m_picea_abies.zip", "aerial_60m_fagus_sylvatica.zip", "aerial_60m_pinus_sylvestris.zip",
    "aerial_60m_quercus_robur.zip", "aerial_60m_pinus_nigra.zip", "aerial_60m_pinus_strobus.zip",
]  # DECISIONS D6: 7.62 GB


def record_files() -> dict:
    if not RECORD_JSON.exists():
        RECORD_JSON.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/api/records/{RECORD_ID}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "treesat-genus-study/0.1"}), timeout=60) as r:
            body = r.read()
        RECORD_JSON.write_bytes(body)
        ledger_add(RECORD_JSON.name, len(body), "metadata")
    meta = json.loads(RECORD_JSON.read_text())
    return {f["key"]: f for f in meta["files"]}


def fetch(key: str, info: dict) -> bool:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / key
    size = int(info["size"])
    want_md5 = info["checksum"].split(":", 1)[1]
    if dest.exists() and dest.stat().st_size == size:
        if md5_of(dest) == want_md5:
            print(f"{key}: already complete, md5 ok")
            return True
        print(f"{key}: complete size but md5 mismatch, deleting")
        dest.unlink()
    have = dest.stat().st_size if dest.exists() else 0
    need = size - have
    used = ledger_total()
    if used + need > CAP_BYTES:
        print(f"{key}: SKIPPED, would exceed cap ({(used + need) / 1e9:.3f} GB > {CAP_BYTES / 1e9:.1f} GB)")
        return False
    url = f"https://zenodo.org/api/records/{RECORD_ID}/files/{key}/content"
    req = urllib.request.Request(url, headers={"User-Agent": "treesat-genus-study/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    got = 0
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("ab" if have else "wb") as out:
            if have and resp.status != 206:
                raise RuntimeError(f"server ignored Range (status {resp.status})")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
    finally:
        ledger_add(key, got, "resume" if have else "full")
    dt_s = time.time() - t0
    print(f"{key}: received {got / 1e6:.1f} MB in {dt_s:.0f} s ({got / 1e6 / max(dt_s, 1e-3):.1f} MB/s)")
    if dest.stat().st_size != size:
        print(f"{key}: size {dest.stat().st_size} != {size}")
        return False
    ok = md5_of(dest) == want_md5
    print(f"{key}: md5 {'ok' if ok else 'MISMATCH'}")
    if not ok:
        dest.unlink()
    return ok


def main(keys: list[str]) -> int:
    files = record_files()
    failed = []
    for key in keys:
        if key not in files:
            print(f"{key}: not in record")
            failed.append(key)
            continue
        ok = False
        for attempt in (1, 2):
            try:
                ok = fetch(key, files[key])
            except Exception as exc:  # network errors: one retry, resuming
                print(f"{key}: attempt {attempt} failed: {exc!r}")
                ok = False
            if ok:
                break
            if (RAW / key).exists() is False and ledger_total() + files[key]["size"] > CAP_BYTES:
                break
        if not ok:
            failed.append(key)
    print(f"ledger total: {ledger_total() / 1e9:.3f} GB")
    print("FAILED: " + ",".join(failed) if failed else "ALL OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or LEVEL1_FILES))
