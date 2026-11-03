"""Private permissions for server-managed SQLite, uploads, and parsed documents."""

import os
from pathlib import Path


def private_directory(path: Path) -> Path:
    """Create a storage directory and restrict POSIX permissions to its owner."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)
    return path


def private_file(path: Path) -> None:
    """Restrict a managed file to owner read/write on POSIX systems."""
    if os.name == "posix" and path.exists():
        path.chmod(0o600)


def private_tree(path: Path) -> None:
    """Restrict a known API-owned parsed-document directory and its direct files."""
    private_directory(path)
    if os.name == "posix":
        for child in path.iterdir():
            if child.is_file() and not child.is_symlink():
                private_file(child)
