"""Tests for scripts/index_in_docker.ps1.

The script cannot run here (no Docker on this machine), so the tests check
the invariants that the audit raised: the container script path, the
streaming pipe, and the absence of buffered output capture.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "index_in_docker.ps1"


def _read() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_container_script_lives_in_scripts():
    """run_in_container.sh is referenced from scripts/, not .agent/."""
    assert "scripts/run_in_container.sh" in _read()
    assert ".agent/run_in_container.sh" not in _read()


def test_output_streams_to_log_as_it_is_produced():
    """Neither docker compose up -d nor docker run pipes into Out-String."""
    text = _read()
    # Out-String only remains in an explanatory comment, not in a command.
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert "Out-String" not in line, f"Out-String still used in a command: {line!r}"
    assert 'Write-Log "compose: $_"' in text
    assert 'Write-Log "docker: $_"' in text
    # The pipe must be on the same logical line as the command it streams, so
    # the output is written to the log as each line arrives.
    assert "2>&1 | ForEach-Object" in text


def test_success_is_decided_from_lastexitcode():
    text = _read()
    assert "$LASTEXITCODE" in text
    # The old buffered form wrote the captured output back only after the run.
    assert "$dockerOutput.Trim()" not in text
    assert "$composeOutput.Trim()" not in text


def test_script_file_exists():
    assert SCRIPT.is_file()
