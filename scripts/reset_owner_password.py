"""Reset a local owner's password and revoke every active session."""

from __future__ import annotations

import argparse
import getpass
import sys

from rag_agent.api.runtime import Runtime
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import ensure_auth_schema, reset_owner_password
from rag_agent.auth.password import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="owner login name")
    args = parser.parse_args()
    if not args.username.strip() or len(args.username) > 50:
        parser.error("username must contain 1 to 50 non-whitespace characters")

    password = getpass.getpass("New password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm new password: ")
    if not 12 <= len(password) <= 1024:
        parser.error("password must contain 12 to 1024 characters")
    if password != confirmation:
        parser.error("passwords do not match")

    runtime = Runtime(APISettings.from_env())
    try:
        ensure_auth_schema(runtime)
        with runtime.connection() as connection:
            reset = reset_owner_password(connection, args.username.strip(), hash_password(password))
        if not reset:
            print("Could not reset owner password.", file=sys.stderr)
            return 1
    finally:
        runtime.close()

    print("Owner password reset; all existing owner sessions were revoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
