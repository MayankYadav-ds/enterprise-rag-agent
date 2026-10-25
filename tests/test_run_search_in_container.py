"""Tests for scripts/run_search_in_container.sh argv handling.

The container script reads the search command from the SEARCH_COMMAND
environment variable and word-splits it with 'eval set --', so
'exec "$@"' runs it with correct argv. The bug it fixes: passing the whole
command string as one argv element to bash left "$@" empty and the command
tried to execute a file named "python scripts/search.py ...".

These tests run the real shell script with a stubbed pip, so the word-split
behaviour is exercised without Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_search_in_container.sh"

BASH = shutil.which("bash")


def _stub_pip(tmp_path: Path) -> Path:
    """A fake pip that exits 0, so the script's install step is a no-op."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bindir / "pip3").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(bindir / "pip", 0o755)
    os.chmod(bindir / "pip3", 0o755)
    return bindir


def _run(env_extra: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_extra)
    env["PATH"] = str(_stub_pip(tmp_path)) + os.pathsep + env["PATH"]
    bash = shutil.which("bash")
    assert bash is not None, f"bash not found on PATH (looked under {env['PATH']})"
    return subprocess.run(
        [bash, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_search_command_is_word_split(tmp_path):
    # The command is a single shell string in SEARCH_COMMAND; the script must
    # run it as argv, so 'echo hello world' prints two words, not one.
    result = _run({"SEARCH_COMMAND": "echo hello world"}, tmp_path)
    assert result.returncode == 0
    assert "hello world" in result.stdout


def test_search_command_quoted_arg_is_preserved(tmp_path):
    # A quoted argument inside the command string stays one argv element.
    result = _run({"SEARCH_COMMAND": "echo 'one two'"}, tmp_path)
    assert result.returncode == 0
    assert "one two" in result.stdout


def test_no_command_and_no_env_reports_usage(tmp_path):
    result = _run({}, tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()


def test_argv_positional_fallback_still_works(tmp_path):
    # Without SEARCH_COMMAND, the script falls back to "$@" from argv.
    bindir = _stub_pip(tmp_path)
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(SCRIPT), "echo", "positional", "args"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "positional args" in result.stdout


def test_set_eu_propagates_command_exit_code(tmp_path):
    # A command that exits non-zero must surface as the script's exit code.
    # 'false' is a real argv command (unlike 'exit', a built-in that exec can't
    # run), so exec can replace the shell with it.
    result = _run({"SEARCH_COMMAND": "false"}, tmp_path)
    assert result.returncode != 0
