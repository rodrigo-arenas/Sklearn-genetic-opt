# Executable Documentation Examples

This directory contains short, deterministic Python examples used to render
notebook-like outputs into the VitePress documentation.

Run all generated examples:

```bash
python docs-vitepress/scripts/run-python-examples.py
```

Check that committed docs are in sync:

```bash
python docs-vitepress/scripts/run-python-examples.py --check
```

Each example script should:

- keep runtime small enough for CI;
- set explicit Python, NumPy, estimator, and CV seeds;
- write plots to both `docs-vitepress/public/images` and `docs/images` when the
  asset is shared by VitePress and Sphinx docs;
- expose `GENERATED_SNIPPETS` with markdown keyed by `docs-example` markers.
