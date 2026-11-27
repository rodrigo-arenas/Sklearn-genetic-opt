"""Checks for the built site's crawl signals and the release documentation switch."""

from html import unescape
from pathlib import Path
import ast
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
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
    tutorial = "tutorials/budgeted-optimizer-search"
    latest_tutorial = f"{SITE}/versions/latest/{tutorial}"
    stable_tutorial = DOCS / "versions" / RELEASES[-1] / f"{tutorial}.md"
    if stable_tutorial.is_file():
        assert f"{SITE}/stable/{tutorial}" in locations
        assert latest_tutorial not in locations
    else:
        assert latest_tutorial in locations
    # No numbered release URL is advertised: the stable release lives at /stable/.
    assert {url for url in locations if "/versions/" in url} <= {latest_tutorial}
    stable_pages = {
        re.sub(r"(^|/)index$", r"\1", page.relative_to(dist).as_posix().removesuffix(".html"))
        for page in (dist / "stable").rglob("*.html")
    }
    assert f"{SITE}/stable/" in locations and len(stable_pages) > 1
    assert {url for url in locations if "/stable/" in url} == {
        f"{SITE}/{page}" for page in stable_pages
    }

    for url in locations:
        parsed = urlparse(url)
        assert f"{parsed.scheme}://{parsed.netloc}" == SITE
        relative_html = parsed.path.lstrip("/")
        relative_html += "index.html" if parsed.path.endswith("/") else ".html"
        html = (dist / relative_html).read_text(encoding="utf-8")
        assert f'rel="canonical" href="{url}"' in html
        assert 'name="robots" content="noindex' not in html

    latest_html = (dist / "versions" / "latest" / "index.html").read_text(encoding="utf-8")
    # `latest` repeats the stable page, so it consolidates on it instead of noindex.
    assert f'rel="canonical" href="{SITE}/stable/"' in latest_html
    assert 'name="robots" content="noindex' not in latest_html

    for older in RELEASES[:-1]:
        older_html = (dist / "versions" / older / "index.html").read_text(encoding="utf-8")
        assert f'rel="canonical" href="{SITE}/stable/"' in older_html


@pytest.mark.parametrize("version", [*RELEASES, "latest"])
def test_every_copy_of_a_stable_page_canonicalizes_to_its_stable_url(version):
    page = "tutorials/isolation-forest"
    built = "stable" if version == RELEASES[-1] else f"versions/{version}"
    html = (DOCS / ".vitepress" / "dist" / built / f"{page}.html").read_text(encoding="utf-8")
    assert f'rel="canonical" href="{SITE}/stable/{page}"' in html
    assert f'property="og:url" content="{SITE}/stable/{page}"' in html
    assert 'name="robots" content="noindex' not in html
    # The edit link opens the source file, which stays under versions/.
    assert f"/edit/master/docs-vitepress/versions/{version}/{page}.md" in html


def test_stable_release_numbered_urls_are_redirect_stubs():
    dist = DOCS / ".vitepress" / "dist"
    numbered = dist / "versions" / RELEASES[-1]
    pages = sorted(
        page.relative_to(dist / "stable").as_posix() for page in (dist / "stable").rglob("*.html")
    )
    assert "tutorials/isolation-forest.html" in pages
    # Only stubs remain in the old URL space, one per stable page.
    assert (
        sorted(
            path.relative_to(numbered).as_posix() for path in numbered.rglob("*") if path.is_file()
        )
        == pages
    )
    sitemap = (dist / "sitemap.xml").read_text(encoding="utf-8")
    assert f"/versions/{RELEASES[-1]}/" not in sitemap
    for page in pages:
        target = "/stable/" + re.sub(r"(^|/)index$", r"\1", page.removesuffix(".html"))
        html = (numbered / page).read_text(encoding="utf-8")
        assert f'http-equiv="refresh" content="0; url={target}"' in html
        assert f'rel="canonical" href="{SITE}{target}"' in html
        assert 'name="robots" content="noindex,follow"' in html
        assert f'<a href="{target}">' in html
        # A plain static page: the VitePress app would render its 404 page here.
        assert "<script" not in html


def test_built_site_does_not_link_to_the_stable_release_numbered_urls():
    # The client router does not know the stub URLs, so in-site links must use /stable/.
    dist = DOCS / ".vitepress" / "dist"
    stubs = dist / "versions" / RELEASES[-1]
    numbered_link = re.compile(rf"[\"'`]/versions/{re.escape(RELEASES[-1])}/")
    offenders = [
        path.relative_to(dist).as_posix()
        for path in [*dist.rglob("*.html"), *(dist / "assets").rglob("*.js")]
        if not path.is_relative_to(stubs) and numbered_link.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders


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
        r'<a\b[^>]*href="/(?:stable|versions/latest)/tutorials/budgeted-optimizer-search"'
        r"[^>]*>(.*?)</a>",
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
        target = f"/stable/{page}"
        assert re.search(
            rf'<a\b[^>]*href="{re.escape(target)}"[^>]*>{re.escape(label)}</a>', homepage
        ), f"Homepage is missing the link {label!r} to {target}"


def test_homepage_plot_previews_link_to_full_gallery_sections():
    dist = DOCS / ".vitepress" / "dist"
    homepage = (dist / "index.html").read_text(encoding="utf-8")
    gallery = "/stable/examples/plotting-gallery"
    full_plots = (dist / f"{gallery.lstrip('/')}.html").read_text(encoding="utf-8")
    cards = re.findall(
        rf'<a\b[^>]*href="{re.escape(gallery)}#([^"]+)"[^>]*>(.*?)</a>', homepage, re.S
    )
    assert len({anchor for anchor, _ in cards}) == 6
    for anchor, card in cards:
        assert f'id="{anchor}"' in full_plots
        preview = re.search(r'<img\b[^>]*src="([^"]+)"[^>]*>', card)
        assert preview and re.search(r'\balt="[^"]+"', preview[0])
        assert (dist / preview[1].lstrip("/")).is_file()


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
    dist = DOCS / ".vitepress" / "dist"
    for version in RELEASES:
        built = "stable" if version == RELEASES[-1] else f"versions/{version}"
        html = (dist / built / "index.html").read_text(encoding="utf-8")
        title = f"sklearn-genetic-opt {version} Documentation"
        assert f"<title>{title}</title>" in html
        assert re.search(
            rf'<meta name="description" content="Released documentation for sklearn-genetic-opt {version}\.',
            html,
        )
        assert f'property="og:title" content="{title}"' in html
    latest = (dist / "versions" / "latest" / "index.html").read_text(encoding="utf-8")
    assert "latest / dev" in re.search(r"<title>([^<]+)</title>", latest).group(1)


# Pages whose metadata belongs to a separate change; remove each entry once it passes.
SNIPPET_EXEMPT = {
    "examples/plotting-gallery.md",
    "tutorials/budgeted-optimizer-search.md",
}


def _front_matter(page: Path) -> dict[str, str]:
    """Return the single-line ``title`` and ``description`` values of a page."""
    block = re.match(r"---\r?\n(.*?)\r?\n---", page.read_text(encoding="utf-8"), re.S)
    fields = {}
    for line in block.group(1).splitlines():
        key, _, value = line.partition(":")
        if key in ("title", "description"):
            value = value.strip()
            # A plain YAML scalar ends at " #", which would silently cut the text; single-quoted
            # and block scalars would be measured with their YAML syntax included.
            assert value.startswith('"') or " #" not in value, f"{page}: {line}"
            assert not value.startswith(("'", ">", "|")), f"{page}: {line}"
            fields[key] = json.loads(value) if value.startswith('"') else value
    return fields


def test_indexed_pages_have_search_snippet_sized_metadata():
    problems = []
    for version in ("latest", RELEASES[-1]):
        root = DOCS / "versions" / version
        for page in sorted(root.rglob("*.md")):
            relative = page.relative_to(root).as_posix()
            # Numbered landing pages get their metadata from transformPageData.
            if relative in SNIPPET_EXEMPT or (relative == "index.md" and version != "latest"):
                continue
            fields = _front_matter(page)
            title = f"{fields['title']} | sklearn-genetic-opt"
            description = fields["description"]
            if len(title) > 70:
                problems.append(f"{version}/{relative}: title has {len(title)} characters")
            if not 110 <= len(description) <= 160:
                problems.append(f"{version}/{relative}: description has {len(description)}")
    assert not problems, "\n".join(problems)


def test_generated_pages_keep_their_generator_metadata():
    for generator in sorted((DOCS / "examples-src").glob("page_*.py")):
        call = next(
            node
            for node in ast.walk(ast.parse(generator.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Notebook"
        )
        spec = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call.keywords
            if kw.arg in ("path", "title", "description")
        }
        page = DOCS / "versions" / "latest" / spec.pop("path")
        assert _front_matter(page) == spec, generator.name
        assert f"\n# {spec['title']}\n" in page.read_text(encoding="utf-8"), generator.name


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


def _next_release():
    major, minor = map(int, RELEASES[-1].split("."))
    return f"{major}.{minor + 1}"


def test_freeze_keeps_homepage_stable_links_resolvable(tmp_path):
    next_version = _next_release()
    docs = tmp_path / "docs-vitepress"
    (docs / "scripts").mkdir(parents=True)
    shutil.copy2(DOCS / "scripts" / "freeze-version.mjs", docs / "scripts")
    shutil.copytree(DOCS / "versions" / "latest", docs / "versions" / "latest")

    subprocess.run(
        ["node", "scripts/freeze-version.mjs", next_version],
        cwd=docs,
        check=True,
        capture_output=True,
        text=True,
    )
    # The homepage links to /stable/, which renders the newest frozen release.
    home_sources = [DOCS / "index.md", *(DOCS / ".vitepress/theme/components").glob("Home*.vue")]
    links = {
        link
        for source in home_sources
        for link in re.findall(r"/stable/([\w./-]*)", source.read_text(encoding="utf-8"))
    }
    assert {
        "guide/basic-usage",
        "benchmarks/optimizer-catalog",
        "tutorials/optimizer-backends",
        "examples/sklearn-comparison",
        "comparisons/optuna-vs-sklearn-genetic-opt",
    } <= links
    for link in links:
        page = re.sub(r"(^|/)$", r"\1index", link)
        assert (docs / "versions" / next_version / f"{page}.md").is_file(), link
    frozen_index = (docs / "versions" / next_version / "index.md").read_text(encoding="utf-8")
    latest_index = (docs / "versions" / "latest" / "index.md").read_text(encoding="utf-8")
    assert "pip install sklearn-genetic-opt" in frozen_index
    assert "pip install git+" not in frozen_index
    assert "pip install git+" in latest_index


def test_next_freeze_moves_stable_urls_without_manual_edits(tmp_path):
    previous, new = RELEASES[-1], _next_release()
    docs, dist = tmp_path / "docs", tmp_path / "dist"
    sources = [
        f"versions/{new}/index.md",
        f"versions/{new}/guide/basic-usage.md",
        f"versions/{previous}/guide/basic-usage.md",
        f"versions/{previous}/guide/removed.md",
        "versions/latest/index.md",
        "versions/latest/guide/unreleased.md",
    ]
    for source in sources:
        (docs / source).parent.mkdir(parents=True, exist_ok=True)
        (docs / source).write_text("# Page\n", encoding="utf-8")
    # The new release's pages as VitePress writes them after the rewrite.
    for page in ("index.html", "guide/basic-usage.html"):
        (dist / "stable" / page).parent.mkdir(parents=True, exist_ok=True)
        (dist / "stable" / page).write_text("<html></html>", encoding="utf-8")
    # Releases are ordered by number, not as text: 1.10 is newer than 1.9.
    numbered = tmp_path / "numbered"
    for release in ("0.9", "0.10", "1.0", "1.9", "1.10"):
        (numbered / "versions" / release).mkdir(parents=True)

    script = f"""
import * as routes from {json.dumps((DOCS / ".vitepress" / "stable-routes.mjs").as_uri())}
const [docs, dist, site, sources] = {json.dumps([str(docs), str(dist), SITE, sources])}
const versions = routes.discoverReleaseVersions(docs)
const rewrite = routes.stableRewrite(versions[0])
routes.writeStableStubs(dist, site, versions[0])
console.log(JSON.stringify({{
  versions,
  numbered: routes.discoverReleaseVersions({json.dumps(str(numbered))}),
  pages: sources.map((source) => {{
    const page = rewrite(source)
    return [page, routes.stableCounterpart(docs, versions[0], page) ?? null]
  }}),
}}))
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    routed = json.loads(result.stdout)
    assert routed["versions"] == [new, previous]
    assert routed["numbered"] == ["1.10", "1.9", "1.0", "0.10", "0.9"]
    # [rendered page, canonical path when it is not the page's own URL]
    assert routed["pages"] == [
        ["stable/index.md", None],
        ["stable/guide/basic-usage.md", None],
        [f"versions/{previous}/guide/basic-usage.md", "/stable/guide/basic-usage"],
        [f"versions/{previous}/guide/removed.md", None],
        ["versions/latest/index.md", "/stable/"],
        ["versions/latest/guide/unreleased.md", None],
    ]
    for stub, target in (
        ("index.html", "/stable/"),
        ("guide/basic-usage.html", "/stable/guide/basic-usage"),
    ):
        html = (dist / "versions" / new / stub).read_text(encoding="utf-8")
        assert f'http-equiv="refresh" content="0; url={target}"' in html
        assert f'rel="canonical" href="{SITE}{target}"' in html
    assert not (dist / "versions" / previous).exists()


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


_BUILD_DOCS_SCRIPT = Path(__file__).with_name("build_docs.py")


def _build_docs_help_text() -> str:
    """Run the public docs-builder CLI with ``--help`` and return its stdout."""
    proc = subprocess.run(
        [sys.executable, str(_BUILD_DOCS_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_build_docs_check_help_describes_actual_behavior():
    """``build_docs.py --check`` help must describe what the flag actually does.

    ``--check`` executes the selected generators end to end and reports
    failures; it does not byte-diff generated pages against committed
    files, and running it may write generated output. The old help text
    claimed the opposite ("Fail if any committed page differs from the
    generated output"), which implied a read-only diff mode that does
    not exist. Compare whitespace-normalised text so the test does not
    depend on exact line wrapping.
    """
    normalized = " ".join(_build_docs_help_text().split()).lower()

    assert "run the selected generators" in normalized
    assert "report" in normalized and "fail" in normalized
    assert "does not compare generated" in normalized
    assert "may write generated output" in normalized

    # The old, misleading wording must be gone.
    assert "differs from the generated output" not in normalized
    assert "dry-run" not in normalized
    assert "dry run" not in normalized
