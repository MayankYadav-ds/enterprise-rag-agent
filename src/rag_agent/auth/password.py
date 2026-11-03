"""Argon2id password hashing and safe verification."""

from argon2 import PasswordHasher
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
