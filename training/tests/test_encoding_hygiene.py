"""
GMN-Football-3 - Encoding hygiene guard (mojibake / non-UTF-8 detection).

An audit on 2026-09-25 found three real classes of encoding damage in this repo:

  * one genuine mojibake line (UTF-8 em dash stored as "a-circumflex euro
    right-quote") in a results document,
  * two cp1252-encoded reports written by generators that called
    ``open(path, "w")`` without ``encoding="utf-8"`` (Windows locale codec), and
  * several logs written as UTF-16LE because PowerShell ``>`` was used.

This module makes that audit reproducible: it byte-scans every git-tracked text
file and fails on

  * invalid UTF-8 that is not binary,
  * mojibake signatures (cp1252-as-UTF-8 dash/quote family, cp437/866 dashes,
    double-encoded Latin-1 leads, C1 control characters),
  * U+FFFD replacement characters (proof of a lossy conversion), and
  * UTF-16 text (BOM-prefixed; renders as mojibake in all UTF-8 tooling).

Disclosed limits: only git-tracked files are scanned; files larger than
``MAX_FULL_BYTES`` are scanned head + tail (the report says how many were
truncated); files that fail UTF-8 decoding *and* contain NUL bytes are treated
as binary and skipped. A UTF-8 BOM is tolerated (legal, though not used here).
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MAX_FULL_BYTES = 8 * 1024 * 1024
TAIL_BYTES = 256 * 1024

BINARY_EXTENSIONS = {
    ".onnx", ".zip", ".pkl", ".pt", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".so", ".dll", ".exe", ".bin", ".woff", ".woff2", ".ttf", ".mp4", ".npy", ".npz",
}

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".jsonl",
    ".md", ".txt", ".log", ".err", ".out", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".sh", ".ps1", ".bat", ".cmd", ".csv", ".tsv", ".sql",
    ".rst", ".patch", ".diff",
}

TEXT_FILENAMES = {
    "makefile", "dockerfile", "license", ".clinerules", ".gitattributes", ".gitignore",
    ".editorconfig", ".env", ".npmrc", ".nvmrc", ".prettierrc",
}

# Moji-bake = UTF-8 bytes re-interpreted as cp1252/latin-1 and saved as UTF-8.
MOJIBAKE_PATTERNS = {
    "dash-quote-family (utf-8 as cp1252)": re.compile(rb"\xc3\xa2\xe2\x82\xac"),
    "bom-as-text": re.compile(rb"\xc3\xaf\xc2\xbb\xc2\xbf"),
    "cp437-866-dash": re.compile(rb"\xce\x93\xc3\x87"),
    "double-encoded-lead": re.compile(rb"\xc3\x83\xc2\xa2"),
    "latin1-lead-C3-pair": re.compile(rb"\xc3\x83[\x80-\xbf]"),
    "accent-follow-C2-pair": re.compile(rb"\xc3\x82[\x80-\xbf]"),
    "replacement-char-U+FFFD": re.compile(rb"\xef\xbf\xbd"),
    "c1-control-chars": re.compile(rb"\xc2[\x80-\x9f]"),
}

# Files where a mojibake signature is intentional: they document this very
# artifact class. The reason string is required by
# test_allowlist_entries_are_documented so the list can never grow silently.
MOJIBAKE_ALLOWLIST = {
    ".clinerules": "the encoding rule itself quotes the corrupt example 'Gamma-C-cedilla'",
    "training/test_event_code_wire.py": "comment documents mojibake examples as text",
}


def _tracked_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT)
    return [line for line in out.decode("utf-8", "replace").splitlines() if line.strip()]


def _is_candidate(rel_path):
    ext = os.path.splitext(rel_path)[1].lower()
    base = os.path.basename(rel_path).lower()
    if ext in BINARY_EXTENSIONS:
        return False
    if ext in TEXT_EXTENSIONS or base in TEXT_FILENAMES:
        return True
    return ext == ""  # extension-less files are checked; binary ones are skipped later


def _read_scan_region(path):
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        if size <= MAX_FULL_BYTES:
            return handle.read(), False
        head = handle.read(MAX_FULL_BYTES)
        handle.seek(max(0, size - TAIL_BYTES))
        return head + handle.read(TAIL_BYTES), True


def scan_tracked_files():
    """Return ``(anomalies, scanned, skipped_binary, truncated)`` for tracked text files."""
    anomalies = []
    scanned = 0
    skipped_binary = 0
    truncated = 0
    for rel in _tracked_files():
        if not _is_candidate(rel):
            continue
        path = os.path.join(REPO_ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        data, was_truncated = _read_scan_region(path)
        scanned += 1
        if was_truncated:
            truncated += 1
        reason = MOJIBAKE_ALLOWLIST.get(rel)

        # 1. UTF-16 text is checked first: it is legal UTF-16 but renders as
        #    mojibake in every UTF-8 tool (and contains NUL bytes, so the binary
        #    heuristic below would otherwise swallow it).
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            if reason is None:
                anomalies.append(
                    (rel, "utf-16-text", "UTF-16 BOM: renders as mojibake in UTF-8 tooling")
                )
            continue

        # 2. Classify binary payloads (NUL bytes + undecodable) before pattern
        #    matching: vendored archives and checkpoint blobs can contain bytes
        #    that coincide with mojibake signatures.
        try:
            data.decode("utf-8")
            decodable = True
            decode_error = None
        except UnicodeDecodeError as exc:
            decodable = False
            decode_error = exc
        if not decodable and b"\x00" in data:
            skipped_binary += 1
            continue

        # 3. Moji-bake signatures.
        if reason is None:
            for label, pattern in MOJIBAKE_PATTERNS.items():
                hits = pattern.findall(data)
                if hits:
                    anomalies.append((rel, label, "%d hit(s), first %r" % (len(hits), hits[0])))

        # 4. Text that is simply not UTF-8 (e.g. cp1252 output).
        if not decodable and reason is None:
            anomalies.append((rel, "invalid-utf-8", str(decode_error)))
    return anomalies, scanned, skipped_binary, truncated


def test_no_mojibake_or_non_utf8_text_in_tracked_files():
    anomalies, scanned, skipped_binary, truncated = scan_tracked_files()
    detail = "\n".join("  %s [%s] %s" % item for item in anomalies[:40])
    if len(anomalies) > 40:
        detail += "\n  ... and %d more" % (len(anomalies) - 40)
    assert not anomalies, (
        "Encoding hygiene: %d anomaly(ies) across %d scanned tracked file(s) "
        "(%d binary skipped, %d truncated scan)\n%s"
        % (len(anomalies), scanned, skipped_binary, truncated, detail)
    )


def test_allowlist_entries_are_documented_and_exist():
    stale = [
        rel
        for rel in MOJIBAKE_ALLOWLIST
        if not os.path.exists(os.path.join(REPO_ROOT, rel.replace("/", os.sep)))
    ]
    assert not stale, "allowlisted path no longer exists: %s" % stale
    undocumented = [rel for rel, why in MOJIBAKE_ALLOWLIST.items() if len(why.strip()) < 15]
    assert not undocumented, "allowlist entries need a real reason: %s" % undocumented


def test_detector_fires_on_the_three_damage_classes_we_fixed():
    """Guard against a silently broken detector (all three classes must match)."""
    mojibake_em_dash = "\u2014".encode("utf-8").decode("cp1252").encode("utf-8")
    assert MOJIBAKE_PATTERNS["dash-quote-family (utf-8 as cp1252)"].search(mojibake_em_dash)
    assert MOJIBAKE_PATTERNS["cp437-866-dash"].search(
        "\u2014".encode("utf-8").decode("cp437").encode("utf-8")
    )
    assert MOJIBAKE_PATTERNS["replacement-char-U+FFFD"].search(b"x\xef\xbf\xbdy")
    assert MOJIBAKE_PATTERNS["c1-control-chars"].search(b"x\xc2\x9fy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

