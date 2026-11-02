"""Notify Bing of changed sitemap pages after a successful Pages deployment.

Uses only the standard library. The state file records the last accepted build;
keep it between runs to avoid submitting unchanged URLs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

SITE = "https://sklearngeneticopt.rodrigo-arenas.com"
ENDPOINT = "https://www.bing.com/indexnow"
KEY_FILE = "indexnow-key.txt"


def page_path(url, site_dir):
    """Resolve a canonical VitePress URL without escaping the built site."""
    parsed = urlsplit(url)
    if (
        f"{parsed.scheme}://{parsed.netloc}" != SITE
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError(f"Invalid documentation URL: {url}")
    path = unquote(parsed.path).lstrip("/")
    path += "index.html" if parsed.path.endswith("/") else ".html"
    resolved = (site_dir / path).resolve()
    if not resolved.is_relative_to(site_dir.resolve()):
        raise ValueError(f"URL escapes the built site: {url}")
    return resolved


def submit(site_dir, state_path, *, dry_run=False):
    """Submit added, changed and removed URLs; save state only on acceptance."""
    site_dir, state_path = Path(site_dir), Path(state_path)
    key = (site_dir / KEY_FILE).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[a-zA-Z0-9-]{8,128}", key):
        raise ValueError("Invalid IndexNow key")

    root = ET.parse(site_dir / "sitemap.xml").getroot()
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    if root.tag != f"{namespace}urlset":
        raise ValueError("Expected a sitemap urlset")
    current = {}
    for loc in root.findall(f"{namespace}url/{namespace}loc"):
        url = (loc.text or "").strip()
        current[url] = hashlib.sha256(page_path(url, site_dir).read_bytes()).hexdigest()
    if not current:
        raise ValueError("Refusing to submit an empty sitemap")

    previous = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise ValueError("Invalid IndexNow state: expected URL hashes")
        for url, digest in previous.items():
            page_path(url, site_dir)
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Invalid IndexNow state hash: {url}")

    changed = sorted(
        url for url in current.keys() | previous.keys() if current.get(url) != previous.get(url)
    )
    if not changed:
        print("IndexNow: no changed URLs.")
        return

    key_location = f"{SITE}/{KEY_FILE}"
    if not dry_run:
        request = Request(key_location, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=30) as response:
            if response.read().decode("utf-8").strip() != key:
                raise ValueError("Published IndexNow key does not match the deployed build")

    for start in range(0, len(changed), 10_000):
        payload = {
            "host": urlsplit(SITE).netloc,
            "key": key,
            "keyLocation": key_location,
            "urlList": changed[start : start + 10_000],
        }
        if dry_run:
            print(json.dumps(payload, indent=2))
            continue
        request = Request(
            ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            if response.status not in (200, 202):
                raise ValueError(f"Unexpected IndexNow response: HTTP {response.status}")
            pending = " (key validation pending)" if response.status == 202 else ""
            print(
                f"IndexNow: {len(payload['urlList'])} URLs accepted, HTTP {response.status}{pending}."
            )

    if not dry_run:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True, help="Built VitePress site")
    parser.add_argument("--state", type=Path, required=True, help="Last accepted URL hashes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print payloads without network or writes"
    )
    args = parser.parse_args()
    submit(args.site_dir, args.state, dry_run=args.dry_run)
