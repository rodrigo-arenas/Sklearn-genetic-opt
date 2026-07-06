"""Unit tests for the docs link checker (``check_links.py``).

Run with::

    python -m pytest docs-vitepress/scripts/test_check_links.py
"""

import importlib.util
from pathlib import Path

import pytest

# Load the sibling ``check_links.py`` by path (the scripts dir is not a package).
_SCRIPT = Path(__file__).with_name("check_links.py")
_spec = importlib.util.spec_from_file_location("check_links", _SCRIPT)
check_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_links)


@pytest.mark.parametrize(
    "target, expected",
    [
        ("./page", True),
        ("../guide/foo", True),
        ("guide/foo#anchor", True),
        ("", False),
        ("#anchor-only", False),
        ("http://example.com", False),
        ("https://example.com/x", False),
        ("mailto:dev@example.com", False),
        ("//cdn.example.com/x", False),
        ("/images/logo.png", False),
    ],
)
def test_is_internal_relative(target, expected):
    assert check_links._is_internal_relative(target) is expected


def test_resolves(tmp_path):
    # Build a small docs tree.
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "intro.md").write_text("x", encoding="utf-8")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "index.md").write_text("x", encoding="utf-8")
    (tmp_path / "empty").mkdir()  # directory with no index.md
    source = tmp_path / "page.md"
    source.write_text("x", encoding="utf-8")

    # Resolves: VitePress-style (no extension), explicit .md, dir with index.md,
    # and a link that only carries an anchor on an existing page.
    assert check_links._resolves(source, "./guide/intro") is True
    assert check_links._resolves(source, "./guide/intro.md") is True
    assert check_links._resolves(source, "./api") is True
    assert check_links._resolves(source, "./guide/intro#section") is True

    # Broken: missing page, and a directory without an index.md (the edge case).
    assert check_links._resolves(source, "./guide/missing") is False
    assert check_links._resolves(source, "./empty") is False


def test_root_doc_valid_local_link_passes(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "external_references.rst").write_text("x", encoding="utf-8")
    root_doc = tmp_path / "CONTRIBUTING.md"
    root_doc.write_text("[refs](docs/external_references.rst)", encoding="utf-8")
    monkeypatch.setattr(check_links, "ROOT", tmp_path)
    monkeypatch.setattr(check_links, "VERSION_DIRS", [])
    monkeypatch.setattr(check_links, "ROOT_DOCS", [root_doc])
    assert check_links.check() == []


def test_root_doc_missing_local_file_is_reported(tmp_path, monkeypatch):
    root_doc = tmp_path / "CONTRIBUTING.md"
    root_doc.write_text("[refs](docs/missing.rst)", encoding="utf-8")
    monkeypatch.setattr(check_links, "ROOT", tmp_path)
    monkeypatch.setattr(check_links, "VERSION_DIRS", [])
    monkeypatch.setattr(check_links, "ROOT_DOCS", [root_doc])
    assert check_links.check() == [(root_doc, "docs/missing.rst")]


def test_root_doc_external_url_is_skipped(tmp_path, monkeypatch):
    root_doc = tmp_path / "CONTRIBUTING.md"
    root_doc.write_text("[external](https://example.com/missing)", encoding="utf-8")
    monkeypatch.setattr(check_links, "ROOT", tmp_path)
    monkeypatch.setattr(check_links, "VERSION_DIRS", [])
    monkeypatch.setattr(check_links, "ROOT_DOCS", [root_doc])
    assert check_links.check() == []


def test_root_doc_rst_parser_does_not_capture_multiline_prose(tmp_path, monkeypatch):
    root_doc = tmp_path / "README.rst"
    root_doc.write_text(
        "This paragraph mentions <not-a-link> and then continues\\n"
        "across lines without creating a filesystem target.",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_links, "ROOT", tmp_path)
    monkeypatch.setattr(check_links, "VERSION_DIRS", [])
    monkeypatch.setattr(check_links, "ROOT_DOCS", [root_doc])
    assert check_links.check() == []


def test_root_doc_rst_image_directive_broken_same_repo_blob_is_reported(tmp_path, monkeypatch):
    root_doc = tmp_path / "README.rst"
    target = (
        "https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/"
        "docs-vitepress/public/images/missing.png?raw=true"
    )
    root_doc.write_text(f".. image:: {target}\n", encoding="utf-8")
    monkeypatch.setattr(check_links, "ROOT", tmp_path)
    monkeypatch.setattr(check_links, "VERSION_DIRS", [])
    monkeypatch.setattr(check_links, "ROOT_DOCS", [root_doc])
    assert check_links.check() == [(root_doc, target)]
