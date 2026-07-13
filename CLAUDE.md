# sklearn-genetic-opt — Claude Code Instructions

Open-source Python library on PyPI: hyperparameter tuning (`GASearchCV`) and feature selection (`GAFeatureSelectionCV`) for scikit-learn estimators using evolutionary algorithms (DEAP). Python >= 3.12. Docs site: https://sklearngeneticopt.rodrigo-arenas.com/.

## Commands

```bash
pytest sklearn_genetic/                                                        # test suite (tests live inside the package)
pytest sklearn_genetic/ --cov-fail-under=95 --cov=./ --cov-report=term-missing:skip-covered   # coverage gate: 95%
black .                                                                        # formatting (pre-commit hook; CI runs black --check .)
cd docs-vitepress && npm run docs:dev                                          # docs preview
cd docs-vitepress && npm run docs:build                                        # docs build
```

CI: `.github/workflows/ci-tests.yml` (tests), `docs.yml` (VitePress → GitHub Pages on push to `master` and on version tags), `docs-examples.yml`.

## Release checklist

1. Version is dynamic from `sklearn_genetic/_version.py` (`pyproject.toml` uses `dynamic = ["version"]`) — bump it there ONLY.
2. Update `CHANGELOG.md`: move `## Unreleased` content under the new `## X.Y.Z` heading, keeping the section names (`New Features` / `Bug Fixes` / `Breaking Changes`) and PR references (#NNN).
3. Ensure docs in `versions/latest/` are current for the release (see Versioning below — the freeze itself is CI-automated).
4. `python -m build` then `twine check dist/*` must pass.
5. Tag (`v[0-9]*` or `[0-9]*`) triggers the docs deploy and version freeze. NEVER push a release tag without the user's explicit confirmation.

## Code conventions

- Full test coverage for behavior changes (coverage gate is 95%); tests live next to the code under `sklearn_genetic/`.
- Follow the scikit-learn estimator API conventions (fit/predict, `get_params`/`set_params`, sklearn checks) for anything public.
- Public API changes require: docs page update, gallery example if user-facing, and a CHANGELOG entry.
- Black is the only formatter; don't hand-format against it.

## Documentation (VitePress + GitHub Pages)

Source in `docs-vitepress/`. The legacy Sphinx docs have been fully removed — VitePress is the only docs system.

### Versioning (critical)

- **stable** = `docs-vitepress/versions/0.13/` (released tags) · **latest** = `docs-vitepress/versions/latest/` (in development).
- NEVER edit pages under `versions/X.Y/` after that version is released. Only `latest/` changes during development.
- Cutting a new docs version is CI-AUTOMATED: pushing a release tag runs `docs.yml` → `scripts/freeze-version.mjs`, which copies `latest/` to `versions/X.Y/`, strips dev banners, and updates the stable redirect. Version listing is auto-discovered by `discoverReleaseVersions()` in `.vitepress/config.ts` — do NOT hand-copy folders or hand-edit a versions array.

### Page standards

- Front-matter on every page: `title` + `description`.
- VitePress admonitions (`::: info|tip|warning|danger|details`), internal links relative without `.html`, one H1 per page, `python` as the fence language for code blocks.
- Guide/tutorial pages follow: brief intro (what you'll learn) → prerequisites → step-by-step with code → self-contained examples → Tips & Gotchas → Next Steps.
- Style: clear, active voice, examples first, beginner-friendly (no assumed hyperparameter-tuning knowledge).
- Code examples must run without modification: sklearn built-in datasets only (digits, iris, breast_cancer — no CSV downloads), imports at the top of every standalone snippet.
- Feature pages follow the template: Use Cases → How It Works → Configuration (param table: Parameter/Type/Default/Description) → Examples (Basic + Advanced) → Tips & Gotchas → Next Steps.

For doc audits or post-change doc updates use the global `docs-sync` skill (it knows these conventions).
