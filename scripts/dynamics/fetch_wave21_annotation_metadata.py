#!/usr/bin/env python3
"""Restore the small official CALVIN language-annotation metadata for Wave 21.

Purpose
-------
Read the already-recorded ZIP member offsets in the representation fetch
manifest and range-fetch only ``auto_lang_ann.npy`` from the official CALVIN
archive.  No image observations or full dataset archive are downloaded.

Parameters
----------
--manifest: Existing representation metadata fetch manifest.
--split: CALVIN split whose annotation payload is restored (default: training).
--url: Optional archive URL override; otherwise use ``official_url`` in manifest.

Usage
-----
python scripts/dynamics/fetch_wave21_annotation_metadata.py \
  --manifest data/representation/calvin_task_D_D/metadata/fetch_manifest.json \
  --split training

Outputs
-------
Writes the restored NumPy payload to the manifest entry's ``saved_path``, by
default ``data/representation/calvin_task_D_D/metadata/training/``
``lang_annotations/auto_lang_ann.npy``.
"""
from __future__ import annotations

import argparse
import binascii
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Fetch one inclusive byte range and reject a server ignoring Range."""

    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range", "")
    expected = end - start + 1
    if len(payload) != expected or not content_range.startswith(f"bytes {start}-{end}/"):
        raise RuntimeError(
            f"Range request was not honored: bytes={len(payload)}, Content-Range={content_range!r}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--url")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    suffix = f"/{args.split}/lang_annotations/auto_lang_ann.npy"
    matches = [entry for entry in manifest["entries"] if entry["name"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix} entry, found {len(matches)}")
    entry = matches[0]
    url = args.url or manifest["official_url"]
    offset = int(entry["local_header_offset"])

    header = fetch_range(url, offset, offset + 4095)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("Recorded offset does not point to a ZIP local header")
    fields = struct.unpack_from("<HHHHHIIIHH", header, 4)
    name_length, extra_length = fields[-2:]
    data_offset = offset + 30 + name_length + extra_length
    compressed_size = int(entry["compressed_size"])
    compressed = fetch_range(url, data_offset, data_offset + compressed_size - 1)
    method = int(entry["compression_method"])
    if method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif method == 0:
        raw = compressed
    else:
        raise RuntimeError(f"Unsupported ZIP compression method: {method}")
    if len(raw) != int(entry["uncompressed_size"]):
        raise RuntimeError("Uncompressed size differs from the recorded ZIP entry")
    if (binascii.crc32(raw) & 0xFFFFFFFF) != int(entry["crc32"]):
        raise RuntimeError("CRC differs from the official ZIP central-directory record")

    output = Path(entry["saved_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(json.dumps({"output": str(output), "bytes": len(raw), "crc32": int(entry["crc32"])}))


if __name__ == "__main__":
    main()
