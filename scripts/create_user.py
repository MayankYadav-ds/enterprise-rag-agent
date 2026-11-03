"""Bootstrap the first local owner without placing passwords in shell history."""

from __future__ import annotations

import argparse
import getpass
import sqlite3
import sys

from rag_agent.api.runtime import Runtime
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema, user_count
from rag_agent.auth.password import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="initial owner login name")
    parser.add_argument("--admin", action="store_true", help="create the initial owner account")
    args = parser.parse_args()
    if not args.username.strip() or len(args.username) > 50:
        parser.error("username must contain 1 to 50 non-whitespace characters")
    if not args.admin:
        parser.error("owner bootstrap requires --admin")

    runtime = Runtime(APISettings.from_env())
    try:
        ensure_auth_schema(runtime)
        with runtime.connection() as connection:
            if user_count(connection) != 0:
                parser.error("owner bootstrap is only allowed before the first account exists")

        password = getpass.getpass("Password (minimum 12 characters): ")
        confirmation = getpass.getpass("Confirm password: ")
        if not 12 <= len(password) <= 1024:
            parser.error("password must contain 12 to 1024 characters")
        if password != confirmation:
            parser.error("passwords do not match")

        with runtime.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if user_count(connection) != 0:
                parser.error("owner bootstrap is only allowed before the first account exists")
            create_user(
                connection,
                args.username.strip(),
                hash_password(password),
                is_admin=True,
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
