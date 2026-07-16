"""Deterministic, dependency-free USDZ packaging.

A USDZ package is an uncompressed ZIP archive whose first entry is the
default USD layer and whose entry payloads start on 64-byte boundaries.
Apple's ``usdzip`` produces that layout on macOS only; this module writes
the same layout with the standard library so USDZ packaging works on any
platform and rebuilds byte-for-byte for identical input bytes.

Every field of every ZIP record is written explicitly: entries are stored
(never deflated), timestamps are pinned to the DOS epoch (1980-01-01), and
payload alignment uses a well-formed extra field so readers that ignore
unknown extra IDs still parse the archive normally.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

USD_LAYER_SUFFIXES = (".usda", ".usdc", ".usd")

_DOS_EPOCH_TIME = 0
_DOS_EPOCH_DATE = 33  # 1980-01-01: year 0, month 1, day 1.
_ALIGNMENT = 64
# Extra-field ID used for alignment padding; readers skip unknown IDs.  The
# same ID is used by other pure-ZIP USDZ writers such as three.js.
_PAD_EXTRA_ID = 0x1986
_LOCAL_HEADER_SIZE = 30
_CENTRAL_HEADER_SIZE = 46

_ASSET_REFERENCE = re.compile(r"@([^@]+)@")


def referenced_assets(usda: Path) -> list[Path]:
    """Return existing files referenced by relative ``@...@`` paths in a USDA layer.

    Absolute paths and URI-style references are rejected: a reviewed asset
    must only reference siblings so the package cannot capture files from
    elsewhere on the machine.
    """

    text = usda.read_text(encoding="utf-8")
    resolved: list[Path] = []
    seen: set[str] = set()
    for match in _ASSET_REFERENCE.finditer(text):
        reference = match.group(1).strip()
        if not reference or reference in seen:
            continue
        seen.add(reference)
        if reference.startswith(("/", "~")) or "://" in reference or reference.startswith(".."):
            raise ValueError(f"non-local asset reference in {usda.name}: {reference}")
        candidate = usda.parent / reference
        if not candidate.is_file():
            raise FileNotFoundError(f"asset referenced by {usda.name} is missing: {reference}")
        resolved.append(candidate)
    return resolved


def _alignment_extra(header_end: int) -> bytes:
    """Extra-field bytes that place the payload at the next 64-byte boundary."""

    padding = (-header_end) % _ALIGNMENT
    if padding == 0:
        return b""
    if padding < 4:
        padding += _ALIGNMENT
    return struct.pack("<HH", _PAD_EXTRA_ID, padding - 4) + bytes(padding - 4)


def pack_usdz(usdz: Path, files: Sequence[Path]) -> list[str]:
    """Write ``files`` into a deterministic USDZ archive and return entry names.

    The first file must be the USD layer.  Entry names are the file names
    (flat archive), encoded as UTF-8, stored uncompressed, with pinned DOS
    timestamps and 64-byte payload alignment.
    """

    if not files:
        raise ValueError("cannot pack an empty USDZ")
    first = Path(files[0])
    if first.suffix.lower() not in USD_LAYER_SUFFIXES:
        raise ValueError(f"first packaged file must be a USD layer, got: {first.name}")
    names = [Path(item).name for item in files]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate entry names in USDZ: {sorted(names)}")

    body = bytearray()
    central = bytearray()
    for path in (Path(item) for item in files):
        data = path.read_bytes()
        name = path.name.encode("utf-8")
        crc = zlib.crc32(data) & 0xFFFFFFFF
        extra = _alignment_extra(len(body) + _LOCAL_HEADER_SIZE + len(name))
        header_offset = len(body)
        body += struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,  # local file header signature
            20,  # version needed to extract
            0,  # general-purpose flags
            0,  # method: stored
            _DOS_EPOCH_TIME,
            _DOS_EPOCH_DATE,
            crc,
            len(data),
            len(data),
            len(name),
            len(extra),
        )
        body += name + extra + data
        central += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,  # central directory header signature
            (3 << 8) | 20,  # made by: unix, version 2.0
            20,  # version needed to extract
            0,  # general-purpose flags
            0,  # method: stored
            _DOS_EPOCH_TIME,
            _DOS_EPOCH_DATE,
            crc,
            len(data),
            len(data),
            len(name),
            0,  # central extra length
            0,  # comment length
            0,  # disk number
            0,  # internal attributes
            0o100644 << 16,  # external attributes: regular file, 0644
            header_offset,
        )
        central += name

    end_record = struct.pack(
        "<IHHHHIIH",
        0x06054B50,  # end-of-central-directory signature
        0,
        0,
        len(files),
        len(files),
        len(central),
        len(body),
        0,
    )
    temporary = usdz.with_name(f".{usdz.name}.pack.tmp")
    temporary.write_bytes(bytes(body) + bytes(central) + end_record)
    temporary.replace(usdz)
    usdz.chmod(0o644)
    return [Path(item).name for item in files]


def verify_usdz_layout(usdz: Path) -> dict[str, Any]:
    """Check the structural USDZ constraints and return the parsed listing.

    Verifies from the raw bytes that every entry is stored uncompressed,
    every payload starts on a 64-byte boundary, timestamps are pinned to the
    DOS epoch, and the first entry is a USD layer.
    """

    payload = usdz.read_bytes()
    eocd = payload.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("missing ZIP end-of-central-directory record")
    entry_count = struct.unpack_from("<H", payload, eocd + 10)[0]
    central_offset = struct.unpack_from("<I", payload, eocd + 16)[0]
    entries: list[dict[str, Any]] = []
    offset = central_offset
    for _ in range(entry_count):
        if payload[offset : offset + 4] != b"PK\x01\x02":
            raise ValueError("invalid central directory header")
        method, dos_time, dos_date = struct.unpack_from("<HHH", payload, offset + 10)
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", payload, offset + 28)
        header_offset = struct.unpack_from("<I", payload, offset + 42)[0]
        name = payload[offset + _CENTRAL_HEADER_SIZE : offset + _CENTRAL_HEADER_SIZE + name_length].decode("utf-8")
        if payload[header_offset : header_offset + 4] != b"PK\x03\x04":
            raise ValueError(f"invalid local header for {name}")
        local_name_length, local_extra_length = struct.unpack_from("<HH", payload, header_offset + 26)
        data_offset = header_offset + _LOCAL_HEADER_SIZE + local_name_length + local_extra_length
        size = struct.unpack_from("<I", payload, header_offset + 22)[0]
        entries.append(
            {
                "name": name,
                "bytes": int(size),
                "data_offset": int(data_offset),
                "aligned": data_offset % _ALIGNMENT == 0,
                "stored": method == 0,
                "epoch_timestamp": dos_time == _DOS_EPOCH_TIME and dos_date == _DOS_EPOCH_DATE,
            }
        )
        offset += _CENTRAL_HEADER_SIZE + name_length + extra_length + comment_length

    problems = [
        f"{entry['name']}: {issue}"
        for entry in entries
        for issue, ok in (
            ("payload is not 64-byte aligned", entry["aligned"]),
            ("entry is compressed", entry["stored"]),
            ("timestamp is not the DOS epoch", entry["epoch_timestamp"]),
        )
        if not ok
    ]
    if not entries:
        problems.append("archive is empty")
    elif not entries[0]["name"].lower().endswith(USD_LAYER_SUFFIXES):
        problems.append(f"first entry is not a USD layer: {entries[0]['name']}")
    return {"passed": not problems, "problems": problems, "entries": entries}


def package_usdz(usda: Path, usdz: Path) -> dict[str, Any]:
    """Package a USDA layer plus its referenced assets into a verified USDZ."""

    files = [usda, *referenced_assets(usda)]
    entry_names = pack_usdz(usdz, files)
    layout = verify_usdz_layout(usdz)
    if not layout["passed"]:
        raise RuntimeError(f"packaged USDZ failed layout verification: {layout['problems']}")
    return {
        "created": True,
        "packager": "local3d.usdz_pack (pure Python, deterministic)",
        "archive_entries": entry_names,
        "layout": layout,
    }
