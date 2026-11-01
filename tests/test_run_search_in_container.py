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
    """A fake pip that exits 0, so the script's install step is a no-op.

    Note: the script invokes ``python -m pip``, which bypasses PATH, so a pip
    stub on PATH alone cannot intercept the install. Tests that need to prevent
    the install set ``SKIP_EMBEDDINGS_INSTALL=1`` instead; tests that need to
    *observe* whether the install ran install a fake ``python`` that records
    ``-m pip`` invocations.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bindir / "pip3").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(bindir / "pip", 0o755)
    os.chmod(bindir / "pip3", 0o755)
    return bindir


def _stub_python(bindir: Path, marker: Path) -> None:
    """A fake ``python`` that records ``-m pip`` invocations on ``marker``.

    The script calls ``python -m pip install ...``, which bypasses PATH, so a
    pip stub alone cannot intercept it. A fake ``python`` on PATH can, and this
    one writes a marker file when it sees ``-m pip`` so tests can assert on
    whether the install was attempted.
    """
    # Use a forward-slash path: the script is executed by bash, where
    # backslashes are escape characters and a Windows drive letter would
    # corrupt the ``touch`` argument.
    marker_posix = str(marker.as_posix())
    (bindir / "python").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        f"  touch {marker_posix}\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(bindir / "python", 0o755)


def _run(env_extra: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_extra)
    env["SKIP_EMBEDDINGS_INSTALL"] = "1"
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
    env["SKIP_EMBEDDINGS_INSTALL"] = "1"
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


def test_install_is_skipped_when_flag_set(tmp_path):
    """SKIP_EMBEDDINGS_INSTALL must prevent the embeddings install from running.

    The script calls ``python -m pip``, which bypasses PATH, so the only way
    to intercept it for a test is a fake ``python`` on PATH. This test installs
    one that records ``-m pip`` invocations and confirms the flag suppresses
    them — the install must not be attempted during argument-handling tests.
    """
    bindir = _stub_pip(tmp_path)
    marker = tmp_path / "pip_was_invoked"
    _stub_python(bindir, marker)
    env = dict(os.environ)
    env["SKIP_EMBEDDINGS_INSTALL"] = "1"
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    env["SEARCH_COMMAND"] = "echo hello world"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "hello world" in result.stdout
    assert not marker.exists(), "pip install was attempted despite SKIP_EMBEDDINGS_INSTALL"


def test_install_runs_when_flag_unset(tmp_path):
    """Without SKIP_EMBEDDINGS_INSTALL, the embeddings install must still run.

    The fake ``python`` records the ``-m pip`` invocation; the test asserts the
    marker appears, so a regression that silently skips the install is caught.
    """
    bindir = _stub_pip(tmp_path)
    marker = tmp_path / "pip_was_invoked"
    _stub_python(bindir, marker)
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    env["SEARCH_COMMAND"] = "echo hello world"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "hello world" in result.stdout
    assert marker.exists(), "embeddings install was skipped when the flag was unset"


def test_install_failure_surfaces(tmp_path):
    """A failing install must not be silently swallowed.

    A fake ``python`` that exits non-zero on ``-m pip`` simulates a broken
    install; the script must surface that failure rather than continuing to
    run the search command.
    """
    bindir = _stub_pip(tmp_path)
    (bindir / "python").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        "  echo 'install failed' >&2\n"
        "  exit 1\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(bindir / "python", 0o755)
    env = dict(os.environ)
    env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    env["SEARCH_COMMAND"] = "echo hello world"
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "install failure must surface as non-zero exit"
    assert "install failed" in result.stderr
