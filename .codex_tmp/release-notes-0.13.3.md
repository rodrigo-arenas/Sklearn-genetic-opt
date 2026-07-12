# sklearn-genetic-opt 0.13.3

This release focuses on making search-space authoring easier, checkpointing safer, errors clearer, and the documentation/release flow cleaner.

## Highlights

- **Search-space conversion helper**: `from_sklearn_space` converts common `RandomizedSearchCV`-style spaces into native `Integer`, `Continuous`, and `Categorical` dimensions. It supports list-like categorical choices plus `scipy.stats.randint`, `uniform`, `loguniform`, and `reciprocal` frozen distributions.
- **Estimator presets**: new starter spaces for `RandomForestClassifier`, `RandomForestRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `LogisticRegression`, `SVC`, `XGBClassifier`, and `XGBRegressor`. Presets support `profile="fast"`, `"balanced"`, or `"wide"` and a `prefix` argument for sklearn pipelines.
- **Checkpointing fixes**: `ModelCheckpoint` and resume flows now build constructor-compatible estimator state more reliably, including `GAFeatureSelectionCV` support.
- **Better validation and error messages** across search spaces, plotting helpers, callbacks, MLflow optional dependency handling, and scheduler adapters.
- **Docs cleanup**: VitePress is now the single docs tree for generated images and release-facing docs.

## New Features and Behavior

- Added `from_sklearn_space` for converting sklearn/scipy-style search spaces into native `sklearn-genetic-opt` dimensions.
- Added estimator preset helpers for common classifiers and regressors, including pipeline-friendly prefixes.
- Added `--use-cache` / `--no-use-cache` to `benchmarks/benchmark_fit.py`, threading the option through `GASearchCV` and `GAFeatureSelectionCV` benchmark builders.
- Added clearer successful-save output for `ModelCheckpoint`.
- Improved internal score ranking utilities to handle `NaN` values consistently.

## Bug Fixes

- Fixed `ModelCheckpoint` state generation for `GAFeatureSelectionCV` and removed duplicate `param_grid` state.
- Fixed `random_state=0` handling for `Integer` and `Continuous` dimensions.
- Fixed `Categorical` priors being ignored during sampling.
- Fixed fitness cache restoration during checkpoint resume.
- Fixed clearer validation for `error_score` during estimator construction.
- Fixed and improved several plotting helper errors, including invalid `top_k`, unavailable metrics, unfitted estimators, and missing history fields.
- Improved search-space validation for warm-start configs, unsupported scipy distributions, feature-name count mismatches, preset prefixes, and invalid `param_grid` entries.
- Improved optional dependency messaging when MLflow is not installed.

## Documentation and Maintenance

- Added and expanded documentation for estimator presets, search-space conversion, pipeline preset prefixes, callbacks, troubleshooting, and community articles.
- Added `CITATION.cff` and improved README citation guidance, including a BibTeX example.
- Added internal Markdown link checking for versioned VitePress docs and root docs links.
- Removed legacy `docs/images` duplication; generated figures now live under `docs-vitepress/public/images`.
- Updated release metadata for `0.13.3`.
- Added pre-commit configuration for Black and basic hygiene checks.
- Built and validated release artifacts with `python -m build` and `twine check`.

## Installation

```bash
pip install -U sklearn-genetic-opt==0.13.3
```

## Full Changelog

https://github.com/rodrigo-arenas/Sklearn-genetic-opt/compare/0.13.2...0.13.3

## Contributors

Huge thanks to everyone who contributed code, tests, docs, reviews, and release polish for this version:

@rodrigo-arenas, @mayoka0, @delaidam, @kernelpanic888, @xuu33030, @cc1a2b, @andrianbalanesq, @Manabendu-ai, @milekv, @AndyDLi, @KingSylvan, @Ishita-Agrawal03, @isha-1686, @aastha-m22, @sarkarshrayan2-max, @ShiHuiwen-creat, @acm-rgb, and @jordansilly77-stack.
