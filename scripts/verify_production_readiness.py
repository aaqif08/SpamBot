#!/usr/bin/env python
"""Production-readiness verification for BotShield AI.

Runs the checks that must pass before a release, against the repository (static) and,
optionally, against a running API (``--url``). Exit code 0 = ready, 1 = blockers found.

    python scripts/verify_production_readiness.py            # static checks only
    python scripts/verify_production_readiness.py --url https://api.example.com --strict

Checks
  S1  no demo/mock/sample data in production code  (scripts/check_no_demo_data.py)
  S2  .env files are git-ignored and not tracked; no secrets in frontend VITE_* variables
  S3  no default credentials (admin/admin, demo@example.com, changeme) in production code
  S4  production config template contains every required variable
  S5  Alembic migrations present and head is reachable
  S6  backend test suite passes                              (--skip-tests to skip)
  S7  frontend type-check + unit tests + production build   (--skip-frontend to skip)
  R1  /api/v1/health and /api/v1/health/ready return 200 on the target URL
  R2  security headers and request id present; interactive docs disabled when --strict
  R3  unauthenticated access to protected endpoints is rejected (401)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
PY = str(BACKEND / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
if not Path(PY).exists():
    PY = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_PROD_VARS = [
    "BOTSHIELD_ENVIRONMENT",
    "BOTSHIELD_DATABASE_URL",
    "BOTSHIELD_SECRET_KEY",
    "BOTSHIELD_CORS_ORIGINS",
    "BOTSHIELD_COOKIE_SECURE",
    "BOTSHIELD_STORAGE_BACKEND",
    "BOTSHIELD_JOB_BACKEND",
]
DEFAULT_CRED_PATTERNS = [r"admin@admin", r"admin/admin", r"demo@example\.com", r"password\s*=\s*['\"]admin['\"]", r"changeme(?!_|\w)"]

results: list[tuple[str, bool, str]] = []


def record(code: str, ok: bool, msg: str) -> None:
    results.append((code, ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {code}  {msg}")


def run(cmd: list[str], cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, shell=(os.name == "nt" and cmd[0] in ("npm", "npx")))
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# ---- static checks ---------------------------------------------------------------- #

def s1_no_demo() -> None:
    code, out = run([PY, str(ROOT / "scripts" / "check_no_demo_data.py")], ROOT)
    record("S1", code == 0, "no demo/mock/sample data in production code" if code == 0 else out.strip().splitlines()[-1])


def s2_env_and_secrets() -> None:
    code, tracked = run(["git", "ls-files", ".env", ".env.production", "backend/.env", "frontend/.env"], ROOT)
    tracked_files = [l for l in tracked.split() if l]
    record("S2a", not tracked_files, "no .env files tracked by git" if not tracked_files else f".env tracked: {tracked_files}")
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    record("S2b", ".env" in gi, ".env is listed in .gitignore")
    bad: list[str] = []
    for f in list(FRONTEND.glob(".env*")) + list(FRONTEND.rglob("src/**/*.ts*")):
        if f.name == "node_modules" or "node_modules" in f.parts:
            continue
        for m in re.finditer(r"VITE_[A-Z0-9_]*(SECRET|TOKEN|PASSWORD|KEY)[A-Z0-9_]*", f.read_text(encoding="utf-8", errors="ignore")):
            bad.append(f"{f.relative_to(ROOT)}:{m.group(0)}")
    record("S2c", not bad, "no secrets exposed through VITE_* variables" if not bad else "; ".join(bad))


def s3_default_credentials() -> None:
    hits: list[str] = []
    for f in list((BACKEND / "app").rglob("*.py")) + list((FRONTEND / "src").rglob("*.ts*")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for pat in DEFAULT_CRED_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                hits.append(f"{f.relative_to(ROOT)}: {m.group(0)}")
    record("S3", not hits, "no default credentials in production code" if not hits else "; ".join(hits[:5]))


def s4_prod_template() -> None:
    tpl = (ROOT / ".env.production.example").read_text(encoding="utf-8")
    missing = [v for v in REQUIRED_PROD_VARS if v not in tpl]
    record("S4", not missing, ".env.production.example lists every required variable" if not missing else f"missing: {missing}")


def s5_migrations() -> None:
    versions = list((BACKEND / "alembic" / "versions").glob("*.py"))
    code, out = run([PY, "-m", "alembic", "heads"], BACKEND)
    ok = bool(versions) and code == 0 and "(head)" in out
    record("S5", ok, f"{len(versions)} migration(s), alembic head resolvable" if ok else out.strip()[-300:])


def s6_backend_tests() -> None:
    code, out = run([PY, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"], BACKEND, timeout=3600)
    summary = [l for l in out.strip().splitlines() if "passed" in l or "failed" in l or "error" in l]
    record("S6", code == 0, (summary[-1] if summary else out.strip()[-200:]))


def s7_frontend() -> None:
    for label, cmd in (("S7a tsc", ["npx", "tsc", "-b"]), ("S7b vitest", ["npx", "vitest", "run"]), ("S7c build", ["npx", "vite", "build"])):
        code, out = run(cmd, FRONTEND, timeout=1200)
        record(label.split()[0], code == 0, f"{label.split()[1]} ok" if code == 0 else out.strip()[-300:])


# ---- runtime checks ---------------------------------------------------------------- #

def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def runtime_checks(base: str, strict: bool) -> None:
    base = base.rstrip("/")
    try:
        st, hdr, body = _get(f"{base}/api/v1/health")
    except Exception as e:  # noqa: BLE001
        record("R1", False, f"health unreachable: {e}")
        return
    record("R1a", st == 200, f"/health -> {st}")
    st2, _, body2 = _get(f"{base}/api/v1/health/ready")
    try:
        ready = json.loads(body2 or b"{}")
    except json.JSONDecodeError:
        ready = {}
    record("R1b", st2 == 200, f"/health/ready -> {st2} {ready.get('checks', '')}")
    record("R2a", "x-request-id" in hdr, "X-Request-ID header present")
    record("R2b", hdr.get("x-content-type-options", "").lower() == "nosniff", "X-Content-Type-Options: nosniff")
    if base.startswith("https://"):
        record("R2c", "strict-transport-security" in hdr, "HSTS header present over HTTPS")
    if strict:
        st_docs, _, _ = _get(f"{base}/docs")
        record("R2d", st_docs == 404, f"interactive docs disabled (/docs -> {st_docs})")
    st3, _, _ = _get(f"{base}/api/v1/auth/me")
    record("R3a", st3 == 401, f"/auth/me without token -> {st3}")
    st4, _, _ = _get(f"{base}/api/v1/models")
    record("R3b", st4 == 401, f"/models without token -> {st4}")
    st5, _, _ = _get(f"{base}/api/v1/auth/me", {"Authorization": "Bearer not-a-token"})
    record("R3c", st5 == 401, f"/auth/me with invalid token -> {st5}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="base URL of a running deployment to probe (e.g. https://api.example.com)")
    ap.add_argument("--strict", action="store_true", help="also require interactive docs to be disabled on --url")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-frontend", action="store_true")
    args = ap.parse_args()

    print("BotShield AI — production readiness verification\n")
    s1_no_demo()
    s2_env_and_secrets()
    s3_default_credentials()
    s4_prod_template()
    s5_migrations()
    if not args.skip_tests:
        s6_backend_tests()
    if not args.skip_frontend:
        s7_frontend()
    if args.url:
        runtime_checks(args.url, args.strict)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    print("PRODUCTION_READY:", "YES" if not failed else "NO")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
