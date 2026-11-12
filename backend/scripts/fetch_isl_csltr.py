"""Selectively download files from the ISL-CSLTR corpus zip (Mendeley Data, CC BY 4.0) using HTTP range
requests, so we never pull the full 8.9 GB archive.

Dataset: Elakkiya R & Natarajan B, "ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language
Translation and Recognition", Mendeley Data, V1, 2021. doi:10.17632/kcmpdxky7p.1

    python scripts/fetch_isl_csltr.py --out ../datasets --match Videos_Sentence_Level corpus_csv_files
"""

import argparse
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

URL = (
    "https://data.mendeley.com/public-files/datasets/kcmpdxky7p/files/"
    "c14c8953-be53-4c7a-85d5-5bcd7e79d86e/file_downloaded"
)


def resolve(client: httpx.Client) -> tuple[str, int]:
    r = client.head(URL, follow_redirects=True)
    r.raise_for_status()
    return str(r.url), int(r.headers["content-length"])


def get_range(client: httpx.Client, url: str, a: int, b: int) -> bytes:
    for attempt in range(5):
        try:
            r = client.get(url, headers={"Range": f"bytes={a}-{b}"})
            r.raise_for_status()
            if len(r.content) == b - a + 1:
                return r.content
        except httpx.HTTPError:
            if attempt == 4:
                raise
    raise IOError(f"short read for range {a}-{b}")


def read_index(client: httpx.Client, url: str, size: int) -> list[dict]:
    tail = get_range(client, url, size - 65536, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12 : i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b"PK\x06\x07")
        z64 = struct.unpack("<Q", tail[j + 8 : j + 16])[0]
        cd_size, cd_off = struct.unpack("<QQ", get_range(client, url, z64, z64 + 55)[40:56])
    cd = get_range(client, url, cd_off, cd_off + cd_size - 1)
    p, out = 0, []
    while cd[p : p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        crc, comp, unc = struct.unpack("<III", cd[p + 16 : p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28 : p + 34])
        off = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + nl].decode("utf-8", "replace")
        ex, q = cd[p + 46 + nl : p + 46 + nl + el], 0
        while q < len(ex):
            hid, hl = struct.unpack("<HH", ex[q : q + 4])
            if hid == 1:  # zip64 extra field
                vals, k = list(struct.unpack("<" + "Q" * (hl // 8), ex[q + 4 : q + 4 + hl // 8 * 8])), 0
                if unc == 0xFFFFFFFF:
                    unc, k = vals[k], k + 1
                if comp == 0xFFFFFFFF:
                    comp, k = vals[k], k + 1
                if off == 0xFFFFFFFF:
                    off = vals[k]
            q += 4 + hl
        out.append({"name": name, "method": method, "crc": crc, "comp": comp, "size": unc, "offset": off})
        p += 46 + nl + el + cl
    return out


def fetch(client: httpx.Client, url: str, e: dict, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size == e["size"]:
        return "skip"
    head = get_range(client, url, e["offset"], e["offset"] + 29)
    nl, el = struct.unpack("<HH", head[26:30])
    start = e["offset"] + 30 + nl + el
    data = get_range(client, url, start, start + e["comp"] - 1) if e["comp"] else b""
    if e["method"] == 8:
        data = zlib.decompress(data, -15)
    elif e["method"] != 0:
        raise ValueError(f"unsupported compression {e['method']} for {e['name']}")
    if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
        raise IOError(f"CRC mismatch for {e['name']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../datasets")
    ap.add_argument("--match", nargs="+", default=["Videos_Sentence_Level", "corpus_csv_files", "ISL_CSLRT.txt"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        url, size = resolve(client)
        entries = [
            e
            for e in read_index(client, url, size)
            if not e["name"].endswith("/") and any(m in e["name"] for m in args.match)
        ]
        # small files first so metadata is available early
        entries.sort(key=lambda e: e["comp"])
        total = sum(e["comp"] for e in entries)
        print(f"{len(entries)} files, {total / 1e9:.2f} GB", flush=True)
        done = 0
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(fetch, client, url, e, out / e["name"]): e for e in entries}
            for f in as_completed(futs):
                e = futs[f]
                done += e["comp"]
                try:
                    status = f.result()
                except Exception as ex:  # keep going; rerun the script to retry failures
                    status = f"FAILED {ex!r}"
                print(f"[{done / total * 100:5.1f}%] {status:4s} {e['name']}", flush=True)


if __name__ == "__main__":
    main()
