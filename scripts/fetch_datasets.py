#!/usr/bin/env python
"""Download real, publicly mirrored bot-detection datasets into backend/data/datasets.

1. Cresci-2015 + Cresci-2017 (user level) — the base paper's benchmark datasets.
   Source: MIB project (Cresci et al.), user profiles (``users.csv``) of every
   subset, mirrored as one merged CSV in a public GitHub repository. The
   per-subset row counts match the base paper's Tables 2–3 exactly.
   The tweet files (``tweets.csv``) are NOT part of the public mirror; they are
   distributed by the authors on request. Drop them next to the generated
   ``users.csv`` files and re-import to enable the tweet-derived features.

2. (optional, ``--external``) "Twitter Human Bots" dataset (37k accounts,
   CC BY-SA 3.0, Hugging Face ``airt-ml/twitter-human-bots``) — an independent
   real dataset useful for cross-dataset evaluation.

Usage
-----
    python scripts/fetch_datasets.py                 # Cresci-15 + Cresci-17
    python scripts/fetch_datasets.py --external      # + Twitter Human Bots
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402

CRESCI_MIRROR_URL = "https://raw.githubusercontent.com/dblop/bot_detector_project/HEAD/users_cresci.csv"
EXTERNAL_URL = "https://huggingface.co/datasets/airt-ml/twitter-human-bots/resolve/main/twitter_human_bots_dataset.csv"

# subset file name in the mirror → (target dataset, folder name)
CRESCI_15 = {
    "TFP_users.csv": "TFP",
    "E13_users.csv": "E13",
    "FSF_users.csv": "FSF",
    "INT_users.csv": "INT",
    "TWT_users.csv": "TWT",
}
# The paper's Cresci-17 composition (Table 3): genuine, social spambots 1–3,
# traditional spambots (1,000 accounts = traditional_spambots_1) and fake followers.
CRESCI_17 = {
    "genuine_accounts.csv": "genuine_accounts",
    "social_spambots_1.csv": "social_spambots_1",
    "social_spambots_2.csv": "social_spambots_2",
    "social_spambots_3.csv": "social_spambots_3",
    "traditional_spambots_1.csv": "traditional_spambots_1",
    "fake_followers.csv": "fake_followers",
}
# Present in the mirror but not part of the paper's Table 3 composition.
CRESCI_17_EXTRA = {
    "traditional_spambots_2.csv": "traditional_spambots_2",
    "traditional_spambots_3.csv": "traditional_spambots_3",
    "traditional_spambots_4.csv": "traditional_spambots_4",
}

USER_COLUMNS = [
    "id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "favourites_count",
    "listed_count", "url", "lang", "time_zone", "location", "default_profile", "default_profile_image",
    "geo_enabled", "profile_image_url", "profile_banner_url", "profile_use_background_image",
    "profile_background_image_url_https", "profile_background_tile", "profile_background_image_url",
    "utc_offset", "protected", "verified", "description", "created_at",
]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "BotShield-AI/1.0 (dataset fetch)"})
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    print(f"  saved   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_subsets(df: pd.DataFrame, mapping: dict[str, str], root: Path) -> list[tuple[str, int]]:
    out = []
    for source, folder in mapping.items():
        part = df[df["dataset"] == source]
        if part.empty:
            print(f"  WARNING: subset {source} missing from mirror")
            continue
        cols = [c for c in USER_COLUMNS if c in part.columns]
        target = root / folder / "users.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        part[cols].to_csv(target, index=False)
        out.append((folder, len(part)))
    return out


def fetch_cresci(datasets_dir: Path, raw_dir: Path) -> None:
    raw = download(CRESCI_MIRROR_URL, raw_dir / "users_cresci.csv")
    digest = sha256(raw)
    df = pd.read_csv(raw, low_memory=False, encoding_errors="replace", on_bad_lines="skip")
    df = df[df["dataset"].notna()]
    print(f"  mirror rows: {len(df):,}  sha256: {digest}")

    c15 = write_subsets(df, CRESCI_15, datasets_dir / "cresci-15")
    c17 = write_subsets(df, CRESCI_17, datasets_dir / "cresci-17")
    extra = write_subsets(df, CRESCI_17_EXTRA, datasets_dir / "cresci-17-extra")

    provenance = datasets_dir / "PROVENANCE.md"
    lines = [
        "# Dataset provenance",
        "",
        "## Cresci-2015 / Cresci-2017 (user level)",
        "",
        f"- Source mirror: {CRESCI_MIRROR_URL}",
        f"- SHA-256 of downloaded mirror: `{digest}`",
        "- Original authors / citation:",
        "  - S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, M. Tesconi, \"Fame for sale: Efficient detection of fake Twitter followers\", Decision Support Systems 80 (2015). [Cresci-2015]",
        "  - S. Cresci et al., \"The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race\", WWW Companion (2017). [Cresci-2017]",
        "- Content: account profiles (`users.csv`) only. `tweets.csv` files are distributed by the authors on request "
        "(MIB project) and are not included; tweet-derived features are therefore unavailable until they are added next to each `users.csv`.",
        "- Terms: academic/research use as set by the original authors.",
        "",
        "| Dataset | Subset | Accounts |",
        "|---|---|---|",
    ]
    lines += [f"| cresci-15 | {f} | {n:,} |" for f, n in c15]
    lines += [f"| cresci-17 | {f} | {n:,} |" for f, n in c17]
    lines += [f"| cresci-17-extra (not in paper Table 3) | {f} | {n:,} |" for f, n in extra]
    provenance.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  cresci-15:", ", ".join(f"{f}={n}" for f, n in c15))
    print("  cresci-17:", ", ".join(f"{f}={n}" for f, n in c17))
    print(f"  provenance written to {provenance}")


def fetch_external(datasets_dir: Path, raw_dir: Path) -> None:
    raw = download(EXTERNAL_URL, raw_dir / "twitter_human_bots_dataset.csv")
    df = pd.read_csv(raw, low_memory=False, encoding_errors="replace", on_bad_lines="skip")
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    df = df.rename(columns={"account_type": "label"})
    df["profile_background_tile"] = 0
    df["listed_count"] = 0
    target = datasets_dir / "external" / "twitter_human_bots.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    with (datasets_dir / "PROVENANCE.md").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n## Twitter Human Bots (external)\n\n"
            f"- Source: {EXTERNAL_URL} (Hugging Face `airt-ml/twitter-human-bots`, CC BY-SA 3.0)\n"
            f"- SHA-256: `{sha256(raw)}`\n"
            f"- {len(df):,} real accounts, profile level; labels bot/human. `listed_count` and `profile_background_tile` are not "
            "provided by this dataset and are filled with 0.\n"
        )
    print(f"  external dataset: {len(df):,} rows → {target}")
    print("  label distribution:", df["label"].value_counts().to_dict())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--external", action="store_true", help="Also fetch the Twitter Human Bots dataset")
    p.add_argument("--no-cresci", action="store_true")
    args = p.parse_args()
    settings = get_settings()
    datasets_dir = settings.datasets_dir
    raw_dir = settings.data_dir / "raw"
    if not args.no_cresci:
        print("Cresci-2015 / Cresci-2017")
        fetch_cresci(datasets_dir, raw_dir)
    if args.external:
        print("Twitter Human Bots (external)")
        fetch_external(datasets_dir, raw_dir)
    print("Done. Next: python scripts/train_model.py --dataset cresci-17 --all")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
