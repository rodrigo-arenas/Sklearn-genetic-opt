"""Checks for the built site's crawl signals and the release documentation switch."""

from html import unescape
from pathlib import Path
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

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
    tutorial = "tutorials/budgeted-optimizer-search"
    latest_tutorial = f"{SITE}/versions/latest/{tutorial}"
    stable_tutorial = DOCS / "versions" / RELEASES[-1] / f"{tutorial}.md"
    if stable_tutorial.is_file():
        assert f"{SITE}/versions/{RELEASES[-1]}/{tutorial}" in locations
        assert latest_tutorial not in locations
    else:
        assert latest_tutorial in locations
    assert {url for url in locations if "/versions/latest/" in url} <= {latest_tutorial}

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


def test_budgeted_search_tutorial_links_to_diagnostics():
    latest = DOCS / ".vitepress" / "dist" / "versions" / "latest"
    tutorial = (latest / "tutorials/budgeted-optimizer-search.html").read_text(encoding="utf-8")
    for page, heading in (
        ("api/plots", "plot-fit-work"),
        ("recipes/advanced/checkpointing", "diagnose-a-resume-error"),
    ):
        assert f"{page}#{heading}" in tutorial
        target = (latest / f"{page}.html").read_text(encoding="utf-8")
        assert f'id="{heading}"' in target


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
    tutorial_link = re.search(
        r'<a\b[^>]*href="/versions/[^/]+/tutorials/budgeted-optimizer-search"' r"[^>]*>(.*?)</a>",
        homepage,
        re.DOTALL,
    )
    assert tutorial_link and "Budget and Resume a Search" in tutorial_link[1]
    for page, label in [
        ("guide/basic-usage", "Run Your First Search"),
        ("benchmarks/optimizer-catalog", "Compare Optimizers"),
        ("api/gasearchcv", "GASearchCV API reference"),
        ("api/gafeatureselectioncv", "GAFeatureSelectionCV API reference"),
        ("tutorials/optimizer-backends", "Switch optimization backends"),
        ("benchmarks/optimizer-catalog", "Compare five optimization backends"),
        ("examples/sklearn-comparison", "Compare scikit-learn search methods"),
        ("comparisons/optuna-vs-sklearn-genetic-opt", "Compare Optuna and sklearn-genetic-opt"),
        ("tutorials/tune-xgboost", "XGBoost hyperparameter tuning"),
        ("tutorials/isolation-forest", "Isolation Forest hyperparameter tuning"),
    ]:
        target = f"/versions/{RELEASES[-1]}/{page}"
        assert re.search(
            rf'<a\b[^>]*href="{re.escape(target)}"[^>]*>{re.escape(label)}</a>', homepage
        ), f"Homepage is missing the link {label!r} to {target}"


def test_homepage_metadata_identifies_the_site_and_python_package():
    homepage = (DOCS / ".vitepress" / "dist" / "index.html").read_text(encoding="utf-8")
    title = unescape(re.search(r"<title>([^<]+)</title>", homepage).group(1))
    assert title == "sklearn-genetic-opt | scikit-learn Tuning & Feature Selection"
    for attribute in ('property="og:title"', 'name="twitter:title"'):
        social_title = re.search(rf'<meta {attribute} content="([^"]+)"', homepage).group(1)
        assert unescape(social_title) == title
    assert 'property="og:site_name" content="sklearn-genetic-opt"' in homepage
    graph_text = re.search(r'<script type="application/ld\+json">(.*?)</script>', homepage, re.S)
    graph = json.loads(graph_text.group(1))
    assert graph["@context"] == "https://schema.org"
    nodes = {node["@type"]: node for node in graph["@graph"]}
    site, package = nodes["WebSite"], nodes["SoftwareApplication"]
    assert site["name"] == package["name"] == "sklearn-genetic-opt"
    assert site["url"] == package["url"] == f"{SITE}/"
    assert site["about"]["@id"] == package["@id"]
    assert package["runtimePlatform"] == "Python"
    assert "GASearchCV" in package["description"]
    assert "GAFeatureSelectionCV" in package["description"]
    assert package["installUrl"] == "https://pypi.org/project/sklearn-genetic-opt/"
    assert set(package["sameAs"]) == {
        "https://pypi.org/project/sklearn-genetic-opt/",
        "https://github.com/rodrigo-arenas/Sklearn-genetic-opt",
    }


def test_numbered_docs_landing_describes_the_release_in_search_and_social():
    dist = DOCS / ".vitepress" / "dist" / "versions"
    for version in RELEASES:
        html = (dist / version / "index.html").read_text(encoding="utf-8")
        title = f"sklearn-genetic-opt {version} Documentation"
        assert f"<title>{title}</title>" in html
        assert re.search(
            rf'<meta name="description" content="Released documentation for sklearn-genetic-opt {version}\.',
            html,
        )
        assert f'property="og:title" content="{title}"' in html
    latest = (dist / "latest" / "index.html").read_text(encoding="utf-8")
    assert "latest / dev" in re.search(r"<title>([^<]+)</title>", latest).group(1)


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [("api/GASearchCV", "api/gasearchcv"), ("tutorials/callbacks", "guide/callbacks")],
)
def test_legacy_development_links_reach_equivalent_pages(old_path, new_path):
    dist = DOCS / ".vitepress" / "dist"
    latest = dist / "versions" / "latest"
    target = f"/versions/latest/{new_path}"
    html = (latest / old_path / "index.html").read_text(encoding="utf-8")
    assert f'http-equiv="refresh" content="0; url={target}"' in html
    assert f'<a href="{target}">' in html
    assert f'rel="canonical" href="{SITE}{target}"' in html
    assert (latest / f"{new_path}.html").is_file()
    sitemap = (dist / "sitemap.xml").read_text(encoding="utf-8")
    assert f"{SITE}/versions/latest/{old_path}" not in sitemap


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
    assert f"/versions/{next_version}/benchmarks/optimizer-catalog" in homepage
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
    frozen_paths = (
        docs / ".vitepress" / "theme" / "components" / "HomeLearningPaths.vue"
    ).read_text(encoding="utf-8")
    for target in (
        "tutorials/optimizer-backends",
        "examples/sklearn-comparison",
        "comparisons/optuna-vs-sklearn-genetic-opt",
    ):
        assert f"/versions/{next_version}/{target}" in frozen_paths
        assert (docs / "versions" / next_version / f"{target}.md").is_file()
    frozen_index = (docs / "versions" / next_version / "index.md").read_text(encoding="utf-8")
    latest_index = (docs / "versions" / "latest" / "index.md").read_text(encoding="utf-8")
    assert "pip install sklearn-genetic-opt" in frozen_index
    assert "pip install git+" not in frozen_index
    assert "pip install git+" in latest_index


@pytest.mark.parametrize("script", ["examples:generate", "examples:check"])
def test_example_npm_scripts_reach_the_generator_cli(script):
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if npm is None:
        pytest.skip("npm is not installed")

    # --help is forwarded to the generator, so this checks the entry point without
    # running any example.
    result = subprocess.run(
        [npm, "run", "--silent", script, "--", "--help"],
        cwd=DOCS,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: build_docs.py" in result.stdout


@pytest.fixture
def indexnow_site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "indexnow-key.txt").write_text("a" * 32 + "\n", encoding="utf-8")
    (site / "index.html").write_text("<main>Home</main>", encoding="utf-8")
    (site / "guide.html").write_text("<main>Guide</main>", encoding="utf-8")
    (site / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{SITE}/</loc></url><url><loc>{SITE}/guide</loc></url>"
        "</urlset>",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "submit_indexnow", DOCS / "scripts" / "submit_indexnow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, site, tmp_path / "state.json"


@pytest.mark.parametrize("status", [200, 202])
def test_indexnow_submits_changes_and_removals_only_after_acceptance(
    indexnow_site, monkeypatch, status
):
    module, site, state = indexnow_site
    posts = []

    def respond(request, timeout):
        if request.get_method() == "GET":
            assert request.full_url == f"{SITE}/indexnow-key.txt"
            return io.BytesIO(("a" * 32 + "\n").encode())
        posts.append((request.full_url, json.loads(request.data)))
        response = io.BytesIO()
        response.status = status
        return response

    monkeypatch.setattr(module, "urlopen", respond)
    module.submit(site, state)
    assert posts == [
        (
            "https://www.bing.com/indexnow",
            {
                "host": urlparse(SITE).netloc,
                "key": "a" * 32,
                "keyLocation": f"{SITE}/indexnow-key.txt",
                "urlList": [f"{SITE}/", f"{SITE}/guide"],
            },
        )
    ]
    posts.clear()
    module.submit(site, state)
    assert posts == []

    (site / "guide.html").write_text("<main>Updated</main>", encoding="utf-8")
    module.submit(site, state)
    assert posts[-1][1]["urlList"] == [f"{SITE}/guide"]
    posts.clear()
    (site / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{SITE}/</loc></url></urlset>",
        encoding="utf-8",
    )
    module.submit(site, state)
    assert [payload["urlList"] for _, payload in posts] == [[f"{SITE}/guide"]]
    assert set(json.loads(state.read_text(encoding="utf-8"))) == {f"{SITE}/"}


@pytest.mark.parametrize("failure", ["wrong-key", 403, 429, 500, 204])
def test_indexnow_failure_preserves_last_successful_state(indexnow_site, monkeypatch, failure):
    module, site, state = indexnow_site
    previous = json.dumps({f"{SITE}/": "0" * 64})
    state.write_text(previous, encoding="utf-8")
    posts = []

    def respond(request, timeout):
        if request.get_method() == "GET":
            return io.BytesIO(("wrong" if failure == "wrong-key" else "a" * 32).encode())
        posts.append(request)
        if failure != 204:
            raise HTTPError(request.full_url, failure, "Rejected", {}, io.BytesIO())
        response = io.BytesIO()
        response.status = 204
        return response

    monkeypatch.setattr(module, "urlopen", respond)
    with pytest.raises((HTTPError, ValueError)):
        module.submit(site, state)
    assert state.read_text(encoding="utf-8") == previous
    if failure == "wrong-key":
        assert posts == []


@pytest.mark.parametrize("url", ["https://elsewhere.example/", f"{SITE}/../outside"])
def test_indexnow_rejects_invalid_sitemap_before_network(indexnow_site, monkeypatch, url):
    module, site, state = indexnow_site
    (site / "sitemap.xml").write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{url}</loc></url></urlset>",
        encoding="utf-8",
    )

    def unexpected_network(*args, **kwargs):
        pytest.fail("Invalid sitemap must not reach the network")

    monkeypatch.setattr(module, "urlopen", unexpected_network)
    with pytest.raises(ValueError):
        module.submit(site, state)
    assert not state.exists()


def test_indexnow_dry_run_neither_sends_nor_updates_state(indexnow_site, monkeypatch, capsys):
    module, site, state = indexnow_site

    def unexpected_network(*args, **kwargs):
        pytest.fail("Dry run must not reach the network")

    monkeypatch.setattr(module, "urlopen", unexpected_network)
    module.submit(site, state, dry_run=True)
    assert json.loads(capsys.readouterr().out)["urlList"] == [f"{SITE}/", f"{SITE}/guide"]
    assert not state.exists()


def test_indexnow_batches_removals_and_preserves_state_if_later_batch_fails(
    indexnow_site, monkeypatch
):
    module, site, state = indexnow_site
    previous = json.dumps({f"{SITE}/removed-{index}": "0" * 64 for index in range(10_001)})
    state.write_text(previous, encoding="utf-8")
    batches = []

    def respond(request, timeout):
        if request.get_method() == "GET":
            return io.BytesIO(("a" * 32).encode())
        batches.append(json.loads(request.data)["urlList"])
        if len(batches) == 2:
            raise HTTPError(request.full_url, 429, "Rate limited", {}, io.BytesIO())
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(module, "urlopen", respond)
    with pytest.raises(HTTPError):
        module.submit(site, state)
    assert list(map(len, batches)) == [10_000, 3]
    submitted = [url for batch in batches for url in batch]
    assert len(set(submitted)) == 10_003
    assert set(submitted) == set(json.loads(previous)) | {f"{SITE}/", f"{SITE}/guide"}
    assert state.read_text(encoding="utf-8") == previous


@pytest.mark.parametrize(
    "contents",
    ["[]", '{"https://elsewhere.example/": "bad"}', "{", json.dumps({f"{SITE}/": "bad"})],
)
def test_indexnow_invalid_state_is_not_overwritten(indexnow_site, contents):
    module, site, state = indexnow_site
    state.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        module.submit(site, state, dry_run=True)
    assert state.read_text(encoding="utf-8") == contents


def test_built_site_publishes_indexnow_key():
    key = (DOCS / ".vitepress" / "dist" / "indexnow-key.txt").read_text(encoding="utf-8")
    assert re.fullmatch(r"[a-zA-Z0-9-]{8,128}\n?", key)
    assert key == (DOCS / "public" / "indexnow-key.txt").read_text(encoding="utf-8")
