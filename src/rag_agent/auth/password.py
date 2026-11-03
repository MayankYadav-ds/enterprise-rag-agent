"""Argon2id password hashing and safe verification."""

import secrets
from functools import lru_cache

from argon2 import PasswordHasher, extract_parameters
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a password with Argon2id and a fresh random salt."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    return _PASSWORD_HASHER.hash(password)


def verify_password(hash_: str, password: str) -> bool:
    """Verify only Argon2id hashes; malformed and legacy hashes fail closed."""
    if (
        not isinstance(hash_, str)
        or not hash_.startswith("$argon2id$")
        or not isinstance(password, str)
    ):
        return False
    try:
        return _PASSWORD_HASHER.verify(hash_, password)
    except (InvalidHashError, VerificationError, VerifyMismatchError, TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def dummy_password_hash() -> str:
    """Return a process-local Argon2id hash for constant-work unknown-user checks."""
    return hash_password(secrets.token_urlsafe(32))


def is_argon2id_hash(hash_: str) -> bool:
    """Return whether an encoded value parses as an Argon2id hash."""
    if not isinstance(hash_, str) or not hash_.startswith("$argon2id$"):
        return False
    try:
        return extract_parameters(hash_).type.name == "ID"
    except (InvalidHashError, TypeError, ValueError):
        return False
