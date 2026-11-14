"""Regression tests for the docs builder CLI (``build_docs.py``).

Run with::

    python -m pytest docs-vitepress/scripts/test_build_docs.py
"""

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).with_name("build_docs.py")


def _help_text() -> str:
    """Run the public CLI with ``--help`` and return its stdout."""
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_check_help_describes_actual_behavior():
    """``--check`` help must describe what the flag actually does.

    ``--check`` executes the selected generators end to end and reports
    failures; it does not byte-diff generated pages against committed
    files, and running it may write generated output. The old help text
    claimed the opposite ("Fail if any committed page differs from the
    generated output"), which implied a read-only diff mode that does
    not exist. Compare whitespace-normalised text so the test does not
    depend on exact line wrapping.
    """
    normalized = " ".join(_help_text().split()).lower()

    assert "run the selected generators" in normalized
    assert "report" in normalized and "fail" in normalized
    assert "does not compare generated" in normalized
    assert "may write generated output" in normalized

    # The old, misleading wording must be gone.
    assert "differs from the generated output" not in normalized
    assert "dry-run" not in normalized
    assert "dry run" not in normalized
