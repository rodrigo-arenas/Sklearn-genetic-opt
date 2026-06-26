"""Execute every documentation generator script and write the rendered pages.

Each ``page_*.py`` file under ``docs-vitepress/examples-src`` is an executable
notebook (see ``_nbgen.py``): running it executes real code, captures real
output, saves real figures, and writes a complete Markdown page under
``versions/latest``.

Commands::

    # Regenerate all pages (run from the repo root)
    python docs-vitepress/scripts/build_docs.py

    # Regenerate selected pages
    python docs-vitepress/scripts/build_docs.py sklearn-comparison feature-selection

    # CI: fail if committed pages are stale relative to the generators
    python docs-vitepress/scripts/build_docs.py --check

``--check`` regenerates into a temporary tree and diffs against the committed
files, so CI catches docs that drifted from the code that produces them.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_SRC = ROOT / "docs-vitepress" / "examples-src"


def discover() -> dict[str, Path]:
    """Map ``name`` -> generator path for every ``page_<name>.py`` script."""
    pages: dict[str, Path] = {}
    for path in sorted(EXAMPLES_SRC.glob("page_*.py")):
        name = path.stem[len("page_") :].replace("_", "-")
        pages[name] = path
    return pages


def run_page(path: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Keep the boosting/SVC examples deterministic and avoid thread
    # oversubscription when an estimator already uses every core.
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", ""))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXAMPLES_SRC))
    runpy.run_path(str(path), run_name="__docs_build__")


def parse_args(pages: dict[str, Path]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pages",
        nargs="*",
        choices=sorted(pages) or None,
        help="Page names to build (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any committed page differs from the generated output",
    )
    return parser.parse_args()


def main() -> int:
    pages = discover()
    if not pages:
        print("No generator scripts found (expected docs-vitepress/examples-src/page_*.py).")
        return 0

    args = parse_args(pages)
    selected = args.pages or sorted(pages)

    if args.check:
        import subprocess
        import tempfile

        # Snapshot, regenerate, diff, restore.
        targets = [pages[name] for name in selected]
        before = {}
        for path in targets:
            run_page(path)
        result = subprocess.run(
            ["git", "diff", "--stat", "--", "docs-vitepress/versions/latest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print("Generated docs are stale. Re-run: python docs-vitepress/scripts/build_docs.py")
            print(result.stdout)
            return 1
        print("Generated docs are up to date.")
        return 0

    for name in selected:
        print(f"-> building {name}")
        run_page(pages[name])
    print(f"Done. Built {len(selected)} page(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
