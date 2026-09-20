"""Operational CLI.

    python -m app.cli migrate                 # alembic upgrade head
    python -m app.cli create-admin            # bootstrap organisation + first ADMIN (prompts / env)
    python -m app.cli check-config            # validate settings and connectivity
    python -m app.cli purge-expired-sessions  # housekeeping

No default accounts are ever created automatically. ``create-admin`` reads
credentials from ``BOTSHIELD_ADMIN_EMAIL`` / ``BOTSHIELD_ADMIN_PASSWORD`` /
``BOTSHIELD_ADMIN_ORG`` or prompts interactively.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _settings():
    from app.core.config import get_settings

    return get_settings()


def cmd_migrate(_: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
    command.upgrade(cfg, "head")
    print("Database migrated to head.")
    return 0


def cmd_create_admin(args: argparse.Namespace) -> int:
    from app.db.database import SessionLocal, current_migration_revision
    from app.services.auth_service import create_organization_with_admin, system_has_users

    if current_migration_revision() is None:
        print("Database is not migrated. Run: python -m app.cli migrate", file=sys.stderr)
        return 2
    email = args.email or os.environ.get("BOTSHIELD_ADMIN_EMAIL") or input("Admin email: ").strip()
    org = args.organization or os.environ.get("BOTSHIELD_ADMIN_ORG") or input("Organization name: ").strip()
    password = args.password or os.environ.get("BOTSHIELD_ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Admin password: ")
        if password != getpass.getpass("Confirm password: "):
            print("Passwords do not match", file=sys.stderr)
            return 2
    db = SessionLocal()
    try:
        if system_has_users(db) and not args.additional:
            print("Users already exist. Use --additional to create another organisation.", file=sys.stderr)
            return 3
        organization, user = create_organization_with_admin(db, org, email, password, args.full_name or "")
        print(f"Created organization '{organization.name}' (slug {organization.slug}) and ADMIN user {user.email}.")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        db.close()


def cmd_check_config(_: argparse.Namespace) -> int:
    from app.core.storage import get_storage
    from app.db.database import check_connection, current_migration_revision

    s = _settings()
    ok = True
    print(f"environment       : {s.environment}")
    print(f"database          : {'reachable' if check_connection() else 'UNREACHABLE'} ({s.database_url.split('://')[0]})")
    ok &= check_connection()
    rev = current_migration_revision()
    print(f"migration revision: {rev or 'NOT MIGRATED'}")
    ok &= rev is not None
    try:
        st = get_storage()
        print(f"storage           : {st.describe()}")
    except Exception as exc:  # noqa: BLE001
        print(f"storage           : ERROR {exc}")
        ok = False
    print(f"job backend       : {s.job_backend}" + (f" ({s.redis_url.split('@')[-1]})" if s.redis_url else ""))
    print(f"cors origins      : {s.cors_origin_list}")
    print(f"docs enabled      : {s.docs_enabled}")
    print(f"cookie secure     : {s.cookie_secure}")
    print(f"x_api configured  : {bool(s.x_bearer_token)}")
    print("RESULT            :", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def cmd_purge_sessions(_: argparse.Namespace) -> int:
    from app.db.database import SessionLocal
    from app.db.models import PasswordResetToken, RefreshSession

    now = datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        n1 = db.query(RefreshSession).filter(RefreshSession.expires_at < now).delete()
        n2 = db.query(PasswordResetToken).filter(PasswordResetToken.expires_at < now).delete()
        db.commit()
        print(f"Removed {n1} expired sessions and {n2} expired reset tokens.")
        return 0
    finally:
        db.close()


def cmd_apply_retention(_: argparse.Namespace) -> int:
    from app.db.database import SessionLocal
    from app.services.retention import apply_retention, retention_enabled

    if not retention_enabled():
        print("No retention window configured (BOTSHIELD_RETENTION_PREDICTIONS_DAYS / BOTSHIELD_RETENTION_AUDIT_DAYS). Nothing to do.")
        return 0
    db = SessionLocal()
    try:
        deleted = apply_retention(db)
    finally:
        db.close()
    print(f"Retention applied: {deleted['predictions']} prediction(s), {deleted['audit']} audit row(s) deleted.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="botshield", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    ca = sub.add_parser("create-admin")
    ca.add_argument("--email")
    ca.add_argument("--password", help="prefer BOTSHIELD_ADMIN_PASSWORD or the interactive prompt")
    ca.add_argument("--organization")
    ca.add_argument("--full-name")
    ca.add_argument("--additional", action="store_true", help="create another organisation even if users exist")
    ca.set_defaults(fn=cmd_create_admin)
    sub.add_parser("check-config").set_defaults(fn=cmd_check_config)
    sub.add_parser("purge-expired-sessions").set_defaults(fn=cmd_purge_sessions)
    sub.add_parser("apply-retention").set_defaults(fn=cmd_apply_retention)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
