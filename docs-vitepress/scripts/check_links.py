"""Check internal Markdown links in the versioned VitePress docs.

Scans every Markdown file under the versioned docs trees and verifies that
*relative* links point at a file that exists. VitePress lets you omit the
``.md`` extension and link to a directory's ``index.md``, so this resolver
mirrors that behaviour.

What is checked / skipped:

* Checked: relative links such as ``./other-page`` or ``../guide/foo``
  (with or without a trailing ``#anchor``; the anchor itself is not validated).
* Skipped: external links (``http://``, ``https://``, ``mailto:``), protocol-
  relative ``//`` links, pure anchors (``#section``), and root-absolute links
  (``/images/...`` etc.) which VitePress resolves from ``public/`` separately.

Run (from the repo root)::

    python docs-vitepress/scripts/check_links.py

Exits non-zero and prints every broken link (grouped by source file) when any
internal link cannot be resolved.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_DIRS = [
    ROOT / "docs-vitepress" / "versions" / "latest",
    ROOT / "docs-vitepress" / "versions" / "0.13",
]
ROOT_DOCS = [
    ROOT / "README.rst",
    ROOT / "CONTRIBUTING.md",
]
OWN_REPO_BLOB_PREFIXES = (
    "https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/",
    "https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/main/",
)

# [text](target) and ![alt](target) — capture the target up to whitespace or ')'.
LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)")
# Basic RST inline links and named hyperlink definitions. Targets are kept
# single-token so normal prose with angle brackets is not treated as a path.
RST_LINK_RE = re.compile(r"`[^`<\n]+<([^\s>]+)>`_|^\.\.\s+_[^:]+:\s+(\S+)", re.MULTILINE)
RST_DIRECTIVE_TARGET_RE = re.compile(r"^\.\.\s+(?:image|figure)::\s+(\S+)", re.MULTILINE)

_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "//")


def _is_internal_relative(target: str) -> bool:
    if not target or target.startswith("#"):
        return False  # empty or pure anchor
    if target.startswith(_EXTERNAL_PREFIXES):
        return False  # external
    if target.startswith("/"):
        return False  # root-absolute (public/) — out of scope here
    return True


def _resolves(md_file: Path, target: str) -> bool:
    """Mirror VitePress relative-link resolution (optional .md / index.md)."""
    path_part = target.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return True  # link was only an anchor/query on the same page

    base = (md_file.parent / path_part).resolve()
    candidates = [
        base,  # a direct file link (e.g. an asset)
        base.with_name(base.name + ".md"),  # VitePress drops the .md extension
        base / "index.md",  # a directory served via its index page
    ]
    # ``is_file`` (not ``exists``) so a directory without an ``index.md`` — which
    # VitePress would not serve — is reported as broken.
    return any(candidate.is_file() for candidate in candidates)


def _same_repo_blob_path(target: str) -> str | None:
    for prefix in OWN_REPO_BLOB_PREFIXES:
        if target.startswith(prefix):
            return target[len(prefix) :]
    return None


def _iter_targets(doc_file: Path) -> list[str]:
    text = doc_file.read_text(encoding="utf-8")
    if doc_file.suffix == ".rst":
        targets = []
        for inline, named in RST_LINK_RE.findall(text):
            targets.append(inline or named)
        targets.extend(RST_DIRECTIVE_TARGET_RE.findall(text))
        return targets
    return LINK_RE.findall(text)


def _resolves_root_doc_link(doc_file: Path, target: str) -> bool:
    same_repo = _same_repo_blob_path(target)
    if same_repo is None:
        if not _is_internal_relative(target):
            return True
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        base = (doc_file.parent / path_part).resolve()
    else:
        path_part = same_repo.split("#", 1)[0].split("?", 1)[0]
        base = (ROOT / path_part).resolve()

    try:
        base.relative_to(ROOT)
    except ValueError:
        return False

    candidates = [
        base,
        base.with_name(base.name + ".md"),
        base.with_name(base.name + ".rst"),
        base / "index.md",
    ]
    return any(candidate.is_file() for candidate in candidates)


def check() -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    for version_dir in VERSION_DIRS:
        if not version_dir.exists():
            continue
        for md_file in sorted(version_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            for target in LINK_RE.findall(text):
                if _is_internal_relative(target) and not _resolves(md_file, target):
                    broken.append((md_file, target))
    for root_doc in ROOT_DOCS:
        if not root_doc.exists():
            continue
        for target in _iter_targets(root_doc):
            if not _resolves_root_doc_link(root_doc, target):
                broken.append((root_doc, target))
    return broken


def main() -> int:
    broken = check()
    if broken:
        print(f"Found {len(broken)} broken internal link(s):\n")
        for md_file, target in broken:
            print(f"  {md_file.relative_to(ROOT)} -> {target}")
        return 1
    print("All internal documentation links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
