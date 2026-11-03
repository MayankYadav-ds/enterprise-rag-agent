"""Provision a local application user without placing passwords in shell history."""

from __future__ import annotations

import argparse
import getpass
import sqlite3
import sys

from rag_agent.api.runtime import Runtime
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="unique login name")
    parser.add_argument("--admin", action="store_true", help="grant document recovery access")
    args = parser.parse_args()
    if not args.username.strip() or len(args.username) > 50:
        parser.error("username must contain 1 to 50 non-whitespace characters")

    password = getpass.getpass("Password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if len(password) < 12:
        parser.error("password must contain at least 12 characters")
    if password != confirmation:
        parser.error("passwords do not match")

    runtime = Runtime(APISettings.from_env())
    try:
        ensure_auth_schema(runtime)
        with runtime.connection() as connection:
            create_user(
                connection,
                args.username.strip(),
                hash_password(password),
                is_admin=args.admin,
            )
    except sqlite3.IntegrityError:
        print("Could not create user: username may already exist.", file=sys.stderr)
        return 1
    finally:
        runtime.close()

    print(f"Created user {args.username.strip()!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
