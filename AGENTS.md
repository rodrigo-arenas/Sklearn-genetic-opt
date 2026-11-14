# Agent contribution guide

This is the shared instruction file for AI-assisted contributions throughout this
repository. Read [CONTRIBUTING.md](CONTRIBUTING.md) for contributor setup and issue
coordination, and [SECURITY.md](SECURITY.md) before reporting a vulnerability.
More specific `AGENTS.md` files, if added, apply within their directories.

## Before editing

- Read the issue and relevant code/tests; establish the expected observable result.
  Check issue status and linked PRs before duplicating work. Posting comments or
  contacting others requires the contributor's authorization.
- Inspect `git status` and preserve changes from other contributors or sessions.
  Search for existing helpers, fixtures and conventions before adding new ones.
- Keep one problem per change. Add dependencies, abstractions or configuration only
  for a concrete requirement. Remove code, imports, tests and documentation orphaned
  by the change. Do not hide failures with broad catches or silent defaults.
- Dependency constraints and tool configuration live in `pyproject.toml`; CI jobs
  live in `.github/workflows/`. Check those sources rather than assuming versions.
  Do not require personal skills, absolute machine paths or a particular AI provider.

## Python and scikit-learn contracts

The package provides `GASearchCV` and `GAFeatureSelectionCV` using DEAP.
Changes to either estimator must preserve scikit-learn compatibility:

- Keep constructor parameters compatible with `get_params`, `set_params` and
  `sklearn.base.clone`; preserve fitted attributes and fit/predict/transform contracts.
- For search or evaluation changes, check the affected scoring, CV splitting,
  refit, `error_score` and `cv_results_` behavior. Keep preprocessing and feature
  selection within the training folds; never tune on held-out test data.
- Preserve seeded reproducibility and the intended behavior of caching, duplicate
  candidates, callbacks and repeated fits when touching those paths. Cover both
  estimators when changing their shared implementation.
- Keep optional plotting and MLflow dependencies optional; do not introduce an
  unconditional import into core code without an explicit dependency decision.
- Public behavior changes need updated API documentation, a relevant example when
  useful, and an entry in `CHANGELOG.md`. Preserve compatibility unless the task
  explicitly calls for a breaking change.

## Validation proportional to the change

Run commands from the repository root in an activated Python environment (Python
3.12+). Use the existing environment when suitable. Install only the extras needed;
see `CONTRIBUTING.md` for full setup. Example test environment:

```powershell
python -m pip install -e ".[test,plot,mlflow,lint]"
```

- Every behavior change needs a test of its observable result. For a bug fix, run
  the regression test before the fix and confirm it fails for the intended reason.
  Extend existing tests/fixtures; parametrize related cases rather than copying them.
- Tests live in `sklearn_genetic/tests/` and subpackage `tests/` directories.
  Select affected tests and adjacent flows; for example, a search change can use:

```powershell
python -m pytest sklearn_genetic/tests/test_genetic_search.py -k clone
```

- Use small datasets, populations and generation counts. Seed the search,
  stochastic estimator and CV splitter as applicable; use `n_jobs=1` unless testing
  parallel execution. Assert contracts rather than exact unseeded scores or timing.
- Do not remove assertions, add skips or lower coverage to make a check pass.
  If a test becomes obsolete, explain why. After adding or renaming tests, run any
  repository-wide test-tree guards that exist.
- Check changed Python files with `python -m black --check` followed by their paths.
  Black settings come from `pyproject.toml`. Run installed pre-commit hooks on the
  changed files; never bypass hooks. Review the diff for duplication and dead code.
- Full-suite and coverage runs belong before merge/release or on explicit request,
  not after every edit. CI's coverage gate is 95%; a focused run does not establish
  whole-project coverage. Run heavy checks sequentially:

```powershell
python -m pytest sklearn_genetic/
python -m pytest sklearn_genetic/ --cov-fail-under=95 --cov=./ --cov-report=term-missing:skip-covered
```

- For Markdown-only edits, validate formatting, referenced paths and links; do not
  run the Python suite. For performance changes, use the benchmark procedure in
  `CONTRIBUTING.md` with comparable seeds, environments and options. Report measured
  results honestly, including regressions; do not fabricate or cherry-pick outcomes.

## Documentation changes

- VitePress sources are in `docs-vitepress/`. Edit development documentation under
  `versions/latest/`; numbered `versions/X.Y/` directories are frozen releases.
- Executable pages are generated: read
  [examples-src/README.md](docs-vitepress/examples-src/README.md), edit the relevant
  generator and regenerate the selected pages. Do not hand-edit generated results
  or invent output. Use built-in sklearn datasets and self-contained imports.
- Give pages `title` and `description` front matter and one H1. Use relative internal
  links without `.html`, VitePress admonitions, and `python` fences for Python code.
  Write clear, active English with examples first and explicit prerequisites.
- Guides should progress from introduction and prerequisites through runnable steps
  to tips and next steps. Feature pages should explain use cases, operation,
  configuration (parameter/type/default/description), and basic/advanced examples.
- For site changes, install Node dependencies if needed, then build and check links:

```powershell
npm --prefix docs-vitepress install
npm --prefix docs-vitepress run docs:build
python docs-vitepress/scripts/check_links.py
```

  Changes to docs tooling also need its affected tests under
  `docs-vitepress/scripts/`; see the docs job in `ci-tests.yml`.
- Version freezing is automated by `.github/workflows/docs.yml` and
  `docs-vitepress/scripts/freeze-version.mjs`. Do not manually copy release folders
  or maintain a hard-coded stable version or versions list.

## Delivery and release boundaries

- Before declaring completion, review the final diff and verify the requested
  result. For substantial code changes, obtain an independent review when available
  and resolve material findings; report any review or validation limitation.
- The handoff/PR must state the problem, resulting behavior, exact checks run and
  their outcomes, checks not run, and remaining risks. Never claim an unrun check
  passed. Contributors remain responsible for reviewing AI-generated changes.
- Stage only intended paths and inspect the staged diff. Do not revert unrelated
  work, bypass hooks, force-push, publish or deploy without explicit authorization.
  Do not commit credentials, local agent settings, logs or temporary artifacts.
- Release work only: change the package version in `sklearn_genetic/_version.py`
  (metadata reads it dynamically), move Unreleased changelog entries under the new
  version while retaining section names and PR references, and validate packaging:

```powershell
python -m build
python -m twine check "dist/*"
```

  Run the applicable CI gates before release. Release tags trigger PyPI publishing
  and documentation freezing/deployment; pushing a release tag requires explicit
  authorization.

Keep this file concise and maintained with the commands and contracts it describes.
Add instructions for recurring project-specific mistakes; leave task history and
personal preferences out. Prefer links to authoritative repo files over duplication.
