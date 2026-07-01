import pytest
from pathlib import Path
import smoke_test_examples
from smoke_test_examples import run_smoke_tests, EXAMPLES_SRC


def test_smoke_test_success():
    """Verify that the smoke test runs successfully on a known fast example."""
    failures = run_smoke_tests(["page_plotting_gallery.py"], base_dir=EXAMPLES_SRC)
    assert not failures, f"Expected 0 failures, got {len(failures)}"


def test_smoke_test_failure_reporting(tmp_path):
    """Verify that a failing example correctly reports its failure."""
    failing_script = tmp_path / "page_failing.py"
    failing_script.write_text("raise ValueError('Intentional failure for testing')")

    failures = run_smoke_tests(["page_failing.py"], base_dir=tmp_path)
    assert len(failures) == 1
    filename, exc = failures[0]
    assert filename == "page_failing.py"
    assert isinstance(exc, ValueError)
    assert "Intentional failure for testing" in str(exc)


def test_optional_heavy_dependencies_are_excluded():
    """Verify we don't accidentally add slow XGBoost/LightGBM examples to the defaults."""
    from smoke_test_examples import main
    import ast

    # Read the smoke_test_examples.py source to see what main() uses
    source = Path(__file__).parent / "smoke_test_examples.py"
    tree = ast.parse(source.read_text())

    examples = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "target_examples":
                    if isinstance(node.value, ast.List):
                        examples = [
                            elt.s for elt in node.value.elts if isinstance(elt, ast.Constant)
                        ]

    assert len(examples) > 0
    assert "page_tune_xgboost.py" not in examples
    assert "page_tune_lightgbm.py" not in examples
    assert "page_tune_catboost.py" not in examples
