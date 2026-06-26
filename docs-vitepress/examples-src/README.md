# Executable Documentation Pages

Every documentation page that contains runnable code is generated from a Python
**generator script** in this directory. Running a generator executes real code,
captures the real `stdout` (and the value of a trailing expression, like a
Jupyter cell), saves real matplotlib figures, and writes a complete VitePress
Markdown page under `docs-vitepress/versions/latest/`.

The guarantee: **the code shown in the docs is exactly the code that ran, and
the output shown is exactly what it produced.** There is no separate
"displayed code" that can drift from "executed code".

## Layout

- `_nbgen.py` — the `Notebook` framework (cells, captured output, figures, page writing).
- `page_<name>.py` — one generator per page. The name maps to a page: e.g.
  `page_sklearn_comparison.py` builds `examples/sklearn-comparison.md`.

## Build the pages

Run from the repository root (a Python 3.12+ environment with the package and
its `[all]` extras installed, plus `xgboost lightgbm catboost` for the boosting
tutorials):

```bash
# Regenerate every page
python docs-vitepress/scripts/build_docs.py

# Regenerate selected pages
python docs-vitepress/scripts/build_docs.py sklearn-comparison feature-selection

# CI smoke test: every generator must run end to end (no byte-diff — pages
# embed real wall-clock timings that vary run to run; scores/figures are seeded)
python docs-vitepress/scripts/build_docs.py --check
```

Set thread caps so estimators that use every core (XGBoost, HistGradientBoosting,
…) don't oversubscribe when the search parallelizes candidates:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
```

## Writing a generator

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nbgen import Notebook

nb = Notebook(
    path="examples/my-page.md",
    title="My Page",
    description="One-sentence description for search/preview.",
    intro="Optional intro markdown shown under the H1.",
)

nb.md("Prose written in Markdown.")
nb.code('''
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    print(X.shape)        # captured and rendered as a ```text block
''')
nb.figure("my_plot.png", "Alt text", caption="Optional caption")  # saves current fig
nb.write()
```

Guidelines for every generator:

- Keep runtime small (well under ~2 minutes) — use modest populations/generations
  and moderate dataset sizes.
- Set explicit Python/NumPy/estimator/CV seeds for reproducible output.
- Show the library **favorably and honestly**: the reliable wins are *tuning vs.
  defaults*, *feature selection vs. all features*, and *GA vs. a grid that cannot
  scale*. Never ship a table where the GA looks worse than a baseline; reframe
  against defaults instead of fabricating numbers.
- Figures are written to both `docs-vitepress/public/images/` and `docs/images/`.
