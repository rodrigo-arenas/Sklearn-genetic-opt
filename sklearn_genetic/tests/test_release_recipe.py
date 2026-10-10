"""Exercise the conda recipe updater used by tagged releases."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

UPDATE_SOURCE = Path(__file__).resolve().parents[2] / "conda.recipe" / "update_source.py"
VERSION = "0.14.0"
SHA256 = "a" * 64


@pytest.mark.parametrize(
    "old_source",
    [
        "source:\n  path: ..\n",
        "source:\n  url: https://example.org/old-0.13.2.tar.gz\n  sha256: " + "b" * 64 + "\n",
    ],
)
def test_release_updater_replaces_local_or_previous_release_source(tmp_path, old_source):
    recipe = tmp_path / "meta.yaml"
    recipe.write_text(
        "package:\n  name: sklearn-genetic-opt\n\n" + old_source + "\nbuild:\n  number: 0\n"
    )

    result = subprocess.run(
        [sys.executable, str(UPDATE_SOURCE), VERSION, SHA256, str(recipe)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == (
        "package:\n  name: sklearn-genetic-opt\n\n"
        "source:\n"
        "  url: https://pypi.io/packages/source/s/sklearn-genetic-opt/"
        "sklearn_genetic_opt-0.14.0.tar.gz\n"
        f"  sha256: {SHA256}\n\n"
        "build:\n  number: 0\n"
    )


@pytest.mark.parametrize(
    "original",
    [
        "source:\n  git_url: https://example.org/repo.git\n",
        "source:\n  path: ..\n  url: https://example.org/old.tar.gz\n  sha256: " + "b" * 64 + "\n",
        "source:\n  url: https://example.org/old.tar.gz\n  sha256: "
        + "b" * 64
        + "\n  patches:\n    - fix.patch\n",
    ],
)
def test_release_updater_rejects_unknown_source_without_writing(tmp_path, original):
    recipe = tmp_path / "meta.yaml"
    recipe.write_text(original)

    result = subprocess.run(
        [sys.executable, str(UPDATE_SOURCE), VERSION, SHA256, str(recipe)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unsupported conda recipe source" in result.stderr
    assert recipe.read_text() == original


def test_conda_runtime_dependencies_match_project_metadata():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    core = {dependency.split(">=")[0] for dependency in project["project"]["dependencies"]}
    recipe = (root / "conda.recipe" / "meta.yaml").read_text(encoding="utf-8")
    run_requirements = recipe.split("  run:\n", 1)[1].split("\ntest:\n", 1)[0]
    conda = {
        line.strip().removeprefix("- ").split(" ")[0]
        for line in run_requirements.splitlines()
        if line.strip().startswith("- ")
    }

    assert conda - {"python"} == core
