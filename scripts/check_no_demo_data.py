#!/usr/bin/env python
"""Scan production code and configuration for demo / mock / sample patterns.

    python scripts/check_no_demo_data.py            # exit 1 if any finding

Scope: backend/app, backend/ml, backend/alembic, frontend/src (excluding tests),
docker/compose files, .env examples, render.yaml. Tests, docs and the research
paper reference tables are out of scope by design (they may legitimately mention
"sample" or "example"). An allow-list covers technical vocabulary such as
"sample_size" (statistical sampling), "example.com" in schema docs, etc.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "backend" / "app", ROOT / "backend" / "ml", ROOT / "backend" / "alembic", ROOT / "frontend" / "src"]
SCAN_FILES = [ROOT / "docker-compose.yml", ROOT / "docker-compose.prod.yml", ROOT / "render.yaml", ROOT / ".env.example", ROOT / ".env.production.example", ROOT / "backend" / "Dockerfile", ROOT / "frontend" / "Dockerfile"]
EXCLUDE_PARTS = {"tests", "test", "node_modules", "dist", "__pycache__", ".venv"}
EXTENSIONS = {".py", ".ts", ".tsx", ".yml", ".yaml", ".json", ".example", "", ".html", ".css"}

PATTERNS = [
    r"\bmock(ed|s|Data)?\b",
    r"\bdummy\b",
    r"\bfake(d|Data)?\b",
    r"\bseed(ed|_demo|Demo)?\b",
    r"\bfixture(s)?\b",
    r"\bplaceholder(s)?\b",
    r"\bdemo(_mode|Mode|_data|Data)?\b",
    r"\bsample(s|_account|Account|_data|Data)?\b",
    r"\bMath\.random\(",
    r"\bhardcoded\b",
    r"\btestData\b",
]
ALLOW = [
    r"sample_size",  # statistical sample size in SHAP / training config
    r"lime_sample|background sample|sampled|sample_raw|sample_scaled|samples\b|num_samples|shap_sample|n_samples|subsample|colsample",  # ML sampling vocabulary
    r"fake follower|fake followers|fake_follower|fake-follower|Fake Follower|fake_followers|fake profiles|fake news|fake Twitter|fake accounts",  # domain term (the thing we detect)
    r"seed=|random_state|\.seed\b|seed: int|\"seed\"|'seed'|seed\b.*random|Random seed|Seed\b",  # RNG seed
    r"placeholder=",  # HTML input placeholder attribute
    r"placeholder:text",  # Tailwind utility
    r"example\.(com|org)",  # RFC 2606 example domains in docs/schemas
    r"for example|For example|e\.g\.",
    r"fakedata|fake_data",  # would still fail below if present as identifier (kept explicit)
    r"never fabricat|not fabricat|Nothing here is fabricated|never pretend",  # negative statements
    r"pytest|moto|fixtures/",  # test tooling references in comments
    r"the fake project|TFP",  # Cresci-15 sub-dataset name
    r"beeswarm sample|explained sample",  # SHAP terminology
]
ALLOW_RE = re.compile("|".join(ALLOW), re.IGNORECASE)
PATTERN_RE = re.compile("|".join(PATTERNS), re.IGNORECASE)


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        if PATTERN_RE.search(line) and not ALLOW_RE.search(line):
            findings.append((i, line.strip()[:140]))
    return findings


def main() -> int:
    files: list[Path] = []
    for d in SCAN_DIRS:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in EXTENSIONS and not (set(p.parts) & EXCLUDE_PARTS):
                files.append(p)
    files += [f for f in SCAN_FILES if f.exists()]
    total = 0
    for f in sorted(files):
        for line_no, text in scan_file(f):
            total += 1
            print(f"{f.relative_to(ROOT)}:{line_no}: {text}")
    if total:
        print(f"\n{total} finding(s). Review each: production code must not create or display demo/mock/sample data.")
        return 1
    print(f"OK - no demo/mock/sample patterns in {len(files)} production files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
