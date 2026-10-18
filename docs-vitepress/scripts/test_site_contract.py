"""Checks for the built site's crawl signals and the release documentation switch."""

from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

DOCS = Path(__file__).resolve().parents[1]
SITE = "https://sklearngeneticopt.rodrigo-arenas.com"
RELEASES = sorted(
    (
        path.name
        for path in (DOCS / "versions").iterdir()
        if path.is_dir() and re.fullmatch(r"\d+\.\d+", path.name)
    ),
    key=lambda version: tuple(map(int, version.split("."))),
)


def test_built_site_advertises_only_indexable_pages():
    dist = DOCS / ".vitepress" / "dist"
    robots = (dist / "robots.txt").read_text(encoding="utf-8")
    assert f"Sitemap: {SITE}/sitemap.xml" in robots

    sitemap = ET.parse(dist / "sitemap.xml")
    locations = {
        element.text
        for element in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    }
    assert f"{SITE}/" in locations
    assert f"{SITE}/versions/{RELEASES[-1]}/" in locations
    assert not any(url.endswith("/stable/") for url in locations)
    assert not any("/versions/latest/" in url for url in locations)

    for url in locations:
        parsed = urlparse(url)
        assert f"{parsed.scheme}://{parsed.netloc}" == SITE
        relative_html = parsed.path.lstrip("/")
        relative_html += "index.html" if parsed.path.endswith("/") else ".html"
        html = (dist / relative_html).read_text(encoding="utf-8")
        assert f'rel="canonical" href="{url}"' in html
        assert 'name="robots" content="noindex' not in html

    stable_html = (dist / "stable" / "index.html").read_text(encoding="utf-8")
    latest_html = (dist / "versions" / "latest" / "index.html").read_text(encoding="utf-8")
    assert f'rel="canonical" href="{SITE}/versions/{RELEASES[-1]}/"' in stable_html
    assert 'name="robots" content="noindex,follow"' in stable_html
    # `latest` repeats the stable page, so it consolidates on it instead of noindex.
    assert f'rel="canonical" href="{SITE}/versions/{RELEASES[-1]}/"' in latest_html
    assert 'name="robots" content="noindex' not in latest_html

    for older in RELEASES[:-1]:
        older_html = (dist / "versions" / older / "index.html").read_text(encoding="utf-8")
        assert f'rel="canonical" href="{SITE}/versions/{RELEASES[-1]}/"' in older_html
    assert not any(f"/versions/{older}/" in url for older in RELEASES[:-1] for url in locations)


def test_homepage_links_to_released_pages():
    dist = DOCS / ".vitepress" / "dist"
    sitemap = ET.parse(dist / "sitemap.xml")
    indexed_paths = {
        urlparse(element.text).path
        for element in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    }
    homepage = (dist / "index.html").read_text(encoding="utf-8")
    missing = []
    for href in re.findall(r'<a\b[^>]*href="([^"]+)"', homepage):
        if not href.startswith("/"):
            continue
        path = urlparse(href).path
        if path == "/versions/latest/" or any(  # explicit version menu entries
            path.startswith(f"/versions/{older}/") for older in RELEASES[:-1]
        ):
            continue
        if path not in indexed_paths:
            missing.append(href)
    assert not missing, f"Homepage links are not indexable pages: {missing}"

    assert "The GA finds better hyperparameters than a grid search" not in homepage
    assert "Joint hyperparameter + feature search in one step" not in homepage


def test_freeze_points_homepage_at_new_stable_version(tmp_path):
    old_version = RELEASES[-1]
    major, minor = map(int, old_version.split("."))
    next_version = f"{major}.{minor + 1}"
    docs = tmp_path / "docs-vitepress"
    (docs / "scripts").mkdir(parents=True)
    (docs / "versions").mkdir()
    (docs / "stable").mkdir()
    shutil.copy2(DOCS / "scripts" / "freeze-version.mjs", docs / "scripts")
    shutil.copytree(DOCS / "versions" / "latest", docs / "versions" / "latest")
    shutil.copy2(DOCS / "stable" / "index.md", docs / "stable")
    shutil.copy2(DOCS / "index.md", docs)
    shutil.copytree(
        DOCS / ".vitepress" / "theme" / "components",
        docs / ".vitepress" / "theme" / "components",
    )

    subprocess.run(
        ["node", "scripts/freeze-version.mjs", next_version],
        cwd=docs,
        check=True,
        capture_output=True,
        text=True,
    )
    homepage = (docs / "index.md").read_text(encoding="utf-8")
    assert f"/versions/{next_version}/guide/basic-usage" in homepage
    assert f"/versions/{old_version}/" not in homepage
    for original in (DOCS / ".vitepress" / "theme" / "components").glob("Home*.vue"):
        source = original.read_text(encoding="utf-8")
        frozen = (docs / ".vitepress" / "theme" / "components" / original.name).read_text(
            encoding="utf-8"
        )
        assert f"/versions/{old_version}/" not in frozen
        if f"/versions/{old_version}/" in source:
            assert f"/versions/{next_version}/" in frozen
    assert (docs / "versions" / next_version / "guide" / "basic-usage.md").is_file()
