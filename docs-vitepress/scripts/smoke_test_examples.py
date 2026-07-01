import os
import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_SRC = ROOT / "docs-vitepress" / "examples-src"


def patch_evolution_config():
    from sklearn_genetic import EvolutionConfig

    original_init = EvolutionConfig.__init__

    def mocked_init(self, *args, **kwargs):
        kwargs["population_size"] = min(kwargs.get("population_size", 2), 2)
        kwargs["generations"] = min(kwargs.get("generations", 1), 1)
        if "keep_top_k" in kwargs:
            kwargs["keep_top_k"] = min(kwargs.get("keep_top_k", 1), 1)
        original_init(self, *args, **kwargs)

    EvolutionConfig.__init__ = mocked_init


def run_smoke_tests(target_examples, base_dir=EXAMPLES_SRC):
    patch_evolution_config()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(base_dir))
    os.environ["MPLBACKEND"] = "Agg"

    failures = []
    for filename in target_examples:
        path = base_dir / filename
        print(f"-> smoke testing {filename}")
        try:
            runpy.run_path(str(path), run_name="__docs_build__")
        except Exception as exc:
            print(f"   FAILED: {filename} - {type(exc).__name__}: {exc}")
            failures.append((filename, exc))
    return failures


def main():
    target_examples = [
        "page_basic_usage.py",
        "page_tutorial_feature_selection.py",
        "page_plotting_gallery.py",
    ]
    failures = run_smoke_tests(target_examples)
    if failures:
        print(f"\n{len(failures)} example(s) failed smoke test.")
        sys.exit(1)
    print(f"\nAll {len(target_examples)} example(s) passed smoke test successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
