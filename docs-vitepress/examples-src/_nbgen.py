"""Lightweight executable-notebook framework for the VitePress docs.

Each documentation page that contains runnable code is authored as a small
Python *generator script* (``page_*.py``) that builds the page with a
``Notebook`` object.  The framework executes every code cell in a shared
namespace, captures the real ``stdout`` (and the value of a trailing
expression), saves any matplotlib figures, and writes a complete Markdown page.

The key guarantee: **the code shown in the docs is exactly the code that ran,
and the output shown is exactly what it produced.**  There is no separate
"displayed code" that can drift from "executed code".

Usage inside a generator script::

    from _nbgen import Notebook

    nb = Notebook(
        path="examples/sklearn-comparison.md",
        title="Comparing Search Methods",
        description="GASearchCV vs RandomizedSearchCV vs GridSearchCV.",
    )
    nb.md("Intro prose written in Markdown.")
    nb.code('''
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        print(X.shape)
    ''')                      # runs it, shows code + captured output
    nb.figure("my_plot.png", "Alt text", caption="Optional caption")
    nb.write()

Run all pages with ``python docs-vitepress/scripts/build_docs.py`` and verify
they are in sync in CI with ``--check``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs-vitepress"
LATEST_ROOT = DOCS_ROOT / "versions" / "latest"
# Images are written to both the VitePress public folder and the legacy Sphinx
# tree so a single run keeps every doc site in sync.
IMAGE_DIRS = [DOCS_ROOT / "public" / "images", ROOT / "docs" / "images"]

DEV_BANNER = (
    ":::warning Development version\n"
    "This is the **latest (dev)** documentation. It may contain unreleased "
    "features or breaking changes. For the stable release, use "
    "[version 0.13](/versions/0.13/).\n"
    ":::"
)


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


class Notebook:
    """Build one Markdown documentation page from executable code cells."""

    def __init__(
        self,
        *,
        path: str,
        title: str,
        description: str,
        dev_banner: bool = True,
        intro: str | None = None,
    ) -> None:
        self.path = LATEST_ROOT / path
        self.title = title
        self.description = description
        self._blocks: list[str] = []
        self.ns: dict[str, object] = {"__name__": "__docs_example__"}

        front = f"---\ntitle: {title}\ndescription: {description}\n---"
        self._blocks.append(front)
        if dev_banner:
            self._blocks.append(DEV_BANNER)
        self._blocks.append(f"# {title}")
        if intro:
            self.md(intro)

    # -- content helpers ---------------------------------------------------
    def md(self, text: str) -> "Notebook":
        """Append a block of Markdown prose."""
        self._blocks.append(_dedent(text))
        return self

    def setup(self, src: str) -> "Notebook":
        """Run hidden setup code (not shown, output discarded)."""
        self._exec(_dedent(src))
        return self

    def code(
        self,
        src: str,
        *,
        echo: bool = True,
        output: bool = True,
        output_lang: str = "text",
    ) -> "Notebook":
        """Show ``src``, run it, and append its captured output.

        The value of a trailing expression (e.g. a bare ``df`` on the last
        line, just like a Jupyter cell) is rendered after any printed output.
        """
        src = _dedent(src)
        if echo:
            self._blocks.append(f"```python\n{src}\n```")
        captured = self._exec(src)
        if output and captured:
            self._blocks.append(f"```{output_lang}\n{captured}\n```")
        return self

    def figure(
        self,
        name: str,
        alt: str,
        *,
        caption: str | None = None,
        dpi: int = 140,
        fig=None,
    ) -> "Notebook":
        """Save the current (or given) figure to the image dirs and embed it."""
        target_fig = fig if fig is not None else plt.gcf()
        for directory in IMAGE_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            target_fig.savefig(directory / name, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        block = f"![{alt}](/images/{name})"
        if caption:
            block += f"\n\n*{caption}*"
        self._blocks.append(block)
        return self

    # -- execution ---------------------------------------------------------
    def _exec(self, src: str) -> str:
        """Execute ``src`` in the shared namespace, returning rendered output."""
        tree = ast.parse(src)
        trailing_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing_expr = tree.body.pop()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(tree, "<docs-cell>", "exec"), self.ns)
            if trailing_expr is not None:
                value = eval(
                    compile(
                        ast.Expression(trailing_expr.value), "<docs-cell>", "eval"
                    ),
                    self.ns,
                )
                if value is not None:
                    print(_render_value(value))

        return out.getvalue().rstrip("\n")

    # -- output ------------------------------------------------------------
    def render(self) -> str:
        return "\n\n".join(block.strip("\n") for block in self._blocks) + "\n"

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render(), encoding="utf-8")
        return self.path


def _render_value(value: object) -> str:
    """Render a trailing-expression value the way a notebook would."""
    try:
        import pandas as pd

        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.to_string()
    except ImportError:  # pragma: no cover
        pass
    return repr(value)
