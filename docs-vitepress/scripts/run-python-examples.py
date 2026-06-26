"""Run curated Python examples and inject their rendered output into VitePress docs.

The examples are regular Python files under ``docs-vitepress/examples-src``.
Each file exposes ``GENERATED_SNIPPETS`` after execution:

    GENERATED_SNIPPETS = {
        "marker-name": "markdown inserted between matching markers",
    }

Markdown pages opt in with markers like:

    <!-- docs-example:marker-name:start -->
    ...
    <!-- docs-example:marker-name:end -->

Use ``--check`` in CI to verify committed docs are in sync with the generated
example outputs.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs-vitepress"
EXAMPLES_ROOT = DOCS_ROOT / "examples-src"
LATEST_ROOT = DOCS_ROOT / "versions" / "latest"

EXAMPLES = {
    "basic-usage": {
        "source": EXAMPLES_ROOT / "basic_usage.py",
        "pages": [LATEST_ROOT / "guide" / "basic-usage.md"],
    },
}


def marker_bounds(name: str) -> tuple[str, str]:
    return (
        f"<!-- docs-example:{name}:start -->",
        f"<!-- docs-example:{name}:end -->",
    )


def extract_marker(content: str, marker: str) -> tuple[str | None, bool]:
    start, end = marker_bounds(marker)
    start_index = content.find(start)
    end_index = content.find(end)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False

    value_start = start_index + len(start)
    return content[value_start:end_index].strip(), True


def replace_marker(content: str, marker: str, generated: str) -> tuple[str, bool]:
    start, end = marker_bounds(marker)
    start_index = content.find(start)
    end_index = content.find(end)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return content, False

    replacement = f"{start}\n{generated.rstrip()}\n{end}"
    new_content = content[:start_index] + replacement + content[end_index + len(end) :]
    return new_content, True


def run_example(source: Path) -> dict[str, str]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    namespace = runpy.run_path(str(source))
    snippets = namespace.get("GENERATED_SNIPPETS")
    if not isinstance(snippets, dict):
        raise RuntimeError(f"{source} did not define GENERATED_SNIPPETS")
    return {str(key): str(value) for key, value in snippets.items()}


def update_pages(example_name: str, check: bool = False) -> bool:
    spec = EXAMPLES[example_name]
    snippets = run_example(spec["source"])
    changed = False

    for page in spec["pages"]:
        original = page.read_text(encoding="utf-8")
        updated = original
        for marker, markdown in snippets.items():
            current, found = extract_marker(updated, marker)
            if not found:
                raise RuntimeError(f"Marker {marker!r} not found in {page}")
            if check and current != markdown.rstrip():
                print(f"{page.relative_to(ROOT)} marker {marker!r} is stale")
            updated, found = replace_marker(updated, marker, markdown)
            if not found:
                raise RuntimeError(f"Marker {marker!r} not found in {page}")

        if updated != original:
            changed = True
            if check:
                print(f"{page.relative_to(ROOT)} is stale")
            else:
                page.write_text(updated, encoding="utf-8")

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        default=sorted(EXAMPLES),
        choices=sorted(EXAMPLES),
        help="Example names to run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated snippets would change committed docs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stale = False
    for example_name in args.examples:
        stale |= update_pages(example_name, check=args.check)

    if args.check and stale:
        print("Run: python docs-vitepress/scripts/run-python-examples.py")
        return 1

    print("Generated docs examples are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
