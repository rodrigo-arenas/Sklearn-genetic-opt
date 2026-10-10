"""Exercise the conda recipe updater used by tagged releases."""

import hashlib
import os
import subprocess
import sys
import textwrap
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


def test_release_workflow_uses_checksum_from_published_sdist(tmp_path):
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "ci-tests.yml").read_text(encoding="utf-8")
    publish = workflow.split("\n  publish:\n", 1)[1].split("\n  sync-conda-recipe:\n", 1)[0]
    sync = workflow.split("\n  sync-conda-recipe:\n", 1)[1]

    assert "sdist_sha256: ${{ steps.sdist_digest.outputs.sha256 }}" in publish
    assert "id: sdist_digest" in publish
    assert publish.index("id: sdist_digest") < publish.index("- name: Publish to PyPI")
    assert "SDIST_SHA256: ${{ needs.publish.outputs.sdist_sha256 }}" in sync
    assert 'echo "SHA256=$SDIST_SHA256" >> "$GITHUB_ENV"' in sync
    assert "python -m build" not in sync

    code = textwrap.dedent(publish.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
    dist = tmp_path / "dist"
    dist.mkdir()
    output = tmp_path / "github-output.txt"
    environment = {**os.environ, "GITHUB_OUTPUT": str(output)}

    def run_digest():
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )

    empty_result = run_digest()
    assert empty_result.returncode != 0
    assert "Expected exactly one sdist, found 0" in empty_result.stderr
    assert not output.exists()

    sdist = dist / "sklearn_genetic_opt-0.14.0.tar.gz"
    sdist.write_bytes(b"published sdist bytes")
    result = run_digest()
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == f"sha256={hashlib.sha256(sdist.read_bytes()).hexdigest()}"

    output.unlink()
    (dist / "another-sdist.tar.gz").write_bytes(b"wrong archive")
    multiple_result = run_digest()
    assert multiple_result.returncode != 0
    assert "Expected exactly one sdist, found 2" in multiple_result.stderr
    assert not output.exists()
