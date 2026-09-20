"""Dataset loading and inspection.

* Cresci-15 / Cresci-17 importer: each subset directory holds ``users.csv`` and
  ``tweets.csv`` (paper §III-C). Tweets are aggregated per user in chunks; text
  derived features (linguistic, sentiment) are computed on up to
  ``max_tweets_per_user`` most recent tweets per account (engineering
  adaptation for tractability — documented in ``docs/methodology.md``).
* Generic CSV inspection for uploads (labelled or unlabelled).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .features import (
    FEATURE_NAMES,
    FeatureExtractor,
    coerce_label,
    detect_label_column,
    normalise_columns,
)
from .preprocessing import count_hashtags, count_mentions, count_urls, text_stats
from .sentiment import average_sentiment
from .utils import json_safe, safe_div, to_float, to_text

log = logging.getLogger(__name__)

# Sub-dataset folder name → (label, human readable category). Matching is
# case-insensitive prefix matching so ``social_spambots_1.csv``-style folders
# and ``TFP``/``E13`` both resolve.
CRESCI_SUBSETS: dict[str, dict[str, tuple[int, str]]] = {
    "cresci-15": {
        "tfp": (0, "Humans (TFP — the fake project)"),
        "e13": (0, "Humans (E13 — elections 2013)"),
        "fsf": (1, "Fake followers (fastfollowerz)"),
        "int": (1, "Fake followers (intertwitter)"),
        "twt": (1, "Fake followers (twittertechnology)"),
    },
    "cresci-17": {
        "genuine_accounts": (0, "Genuine accounts"),
        "social_spambots_1": (1, "Social spambots #1"),
        "social_spambots_2": (1, "Social spambots #2"),
        "social_spambots_3": (1, "Social spambots #3"),
        "traditional_spambots_1": (1, "Traditional spambots #1"),
        "traditional_spambots_2": (1, "Traditional spambots #2"),
        "traditional_spambots_3": (1, "Traditional spambots #3"),
        "traditional_spambots_4": (1, "Traditional spambots #4"),
        "fake_followers": (1, "Fake followers"),
    },
}

#: Reference statistics reported by the base paper (Tables 2 & 3). Shown only as
#: "reported in base paper"; never used as measured statistics.
CRESCI_PAPER_STATS: dict[str, dict[str, Any]] = {
    "cresci-15": {
        "citation": "Cresci et al., 'Fame for sale: Efficient detection of fake Twitter followers', DSS 2015",
        "subsets": [
            {"name": "TFP (the fake project)", "type": "100% humans", "accounts": 469, "tweets": 563693},
            {"name": "E13 (elections 2013)", "type": "100% humans", "accounts": 1481, "tweets": 2068037},
            {"name": "FSF (fastfollowerz)", "type": "100% fake followers", "accounts": 1169, "tweets": 22910},
            {"name": "INT (intertwitter)", "type": "100% fake followers", "accounts": 1337, "tweets": 58925},
            {"name": "TWT (twittertechnology)", "type": "100% fake followers", "accounts": 845, "tweets": 114192},
        ],
    },
    "cresci-17": {
        "citation": "Cresci et al., 'The paradigm-shift of social spambots', WWW Companion 2017",
        "subsets": [
            {"name": "Traditional spambots", "type": "bots", "accounts": 1000, "tweets": 145094},
            {"name": "Social spambots 1", "type": "bots", "accounts": 991, "tweets": 1610176},
            {"name": "Social spambots 2", "type": "bots", "accounts": 3457, "tweets": 428542},
            {"name": "Social spambots 3", "type": "bots", "accounts": 464, "tweets": 1418626},
            {"name": "Fake followers", "type": "bots", "accounts": 3351, "tweets": 196027},
            {"name": "Genuine accounts", "type": "humans", "accounts": 3474, "tweets": 8377522},
        ],
    },
}

TWEET_COLUMNS = ["user_id", "text", "retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"]
USER_COLUMNS = [
    "id", "name", "screen_name", "statuses_count", "followers_count", "friends_count", "favourites_count",
    "listed_count", "url", "location", "default_profile", "default_profile_image", "geo_enabled",
    "profile_image_url", "profile_banner_url", "profile_background_tile", "verified", "description",
]


@dataclass
class SubsetInfo:
    key: str
    label: int
    category: str
    path: Path
    users_file: Path
    tweets_file: Path | None


@dataclass
class CresciScan:
    kind: str
    root: Path
    subsets: list[SubsetInfo] = field(default_factory=list)
    features_cache: Path | None = None

    @property
    def available(self) -> bool:
        return bool(self.subsets)


def _find_file(folder: Path, stem: str) -> Path | None:
    for cand in (folder / f"{stem}.csv", folder / f"{stem}.CSV"):
        if cand.exists():
            return cand
    for p in folder.glob("*.csv"):
        if p.stem.lower() == stem:
            return p
    return None


def scan_cresci(root: Path, kind: str) -> CresciScan:
    """Look for ``<root>/<subset>/users.csv`` (+ ``tweets.csv``) folders."""
    scan = CresciScan(kind=kind, root=root)
    if not root.exists():
        return scan
    cache = root / "features.csv"
    if cache.exists():
        scan.features_cache = cache
    mapping = CRESCI_SUBSETS[kind]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name.lower().replace("-", "_").replace(" ", "_")
        match = next((k for k in mapping if name == k or name.startswith(k)), None)
        if match is None:
            continue
        users = _find_file(folder, "users")
        if users is None:
            continue
        label, category = mapping[match]
        scan.subsets.append(
            SubsetInfo(key=folder.name, label=label, category=category, path=folder, users_file=users, tweets_file=_find_file(folder, "tweets"))
        )
    return scan


def _aggregate_tweets(
    tweets_file: Path,
    max_tweets_per_user: int,
    chunksize: int = 200_000,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Chunked per-user aggregation of a Cresci ``tweets.csv``."""
    sums: dict[str, dict[str, float]] = {}
    texts: dict[str, list[str]] = {}
    total_rows = 0
    header = pd.read_csv(tweets_file, nrows=0, encoding="utf-8", encoding_errors="replace").columns
    usecols = [c for c in TWEET_COLUMNS if c in header]
    if "user_id" not in usecols:
        raise ValueError(f"{tweets_file.name} has no 'user_id' column")
    reader = pd.read_csv(
        tweets_file,
        usecols=usecols,
        chunksize=chunksize,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
        dtype={"user_id": str, "text": str},
    )
    for chunk in reader:
        total_rows += len(chunk)
        chunk["user_id"] = chunk["user_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        has_text = "text" in chunk.columns
        for col in ("retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        if has_text:
            txt = chunk["text"].fillna("")
            if "num_hashtags" not in chunk.columns:
                chunk["num_hashtags"] = txt.map(count_hashtags)
            if "num_urls" not in chunk.columns:
                chunk["num_urls"] = txt.map(count_urls)
            if "num_mentions" not in chunk.columns:
                chunk["num_mentions"] = txt.map(count_mentions)
        for col in ("retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"):
            if col not in chunk.columns:
                chunk[col] = 0.0
        grouped = chunk.groupby("user_id")[["retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"]].sum()
        counts = chunk.groupby("user_id").size()
        for uid, row in grouped.iterrows():
            acc = sums.setdefault(str(uid), {"n_tweets": 0.0, "retweet_count": 0.0, "reply_count": 0.0, "favorite_count": 0.0, "num_hashtags": 0.0, "num_urls": 0.0, "num_mentions": 0.0})
            acc["n_tweets"] += float(counts.get(uid, 0))
            for col in ("retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"):
                acc[col] += float(row[col])
        if has_text:
            for uid, group in chunk.groupby("user_id")["text"]:
                bucket = texts.setdefault(str(uid), [])
                if len(bucket) < max_tweets_per_user:
                    bucket.extend(t for t in group.fillna("").tolist()[: max_tweets_per_user - len(bucket)] if t)
        if progress:
            progress(f"{tweets_file.parent.name}: {total_rows:,} tweets aggregated")

    records = []
    for uid, acc in sums.items():
        bucket = texts.get(uid, [])
        stats = text_stats(bucket)
        sent = average_sentiment(bucket)
        records.append(
            {
                "id": uid,
                "n_tweets": acc["n_tweets"],
                "tweets_observed": acc["n_tweets"],
                "retweet_count": acc["retweet_count"],
                "reply_count": acc["reply_count"],
                "favorite_count_received": acc["favorite_count"],
                "hashtag_count": acc["num_hashtags"],
                "url_count": acc["num_urls"],
                "mentions_count": acc["num_mentions"],
                "unique_word_count": float(stats.unique_words),
                "unique_word_use": float(stats.unique_word_use),
                "punctuation_count": float(stats.punctuation_count),
                "avg_sentence_length": float(stats.avg_sentence_length),
                "punctuation_density": float(stats.punctuation_density),
                "avg_polarity": float(sent.polarity),
                "avg_subjectivity": float(sent.subjectivity),
                "text_tweets_used": float(len(bucket)),
            }
        )
    return pd.DataFrame(records)


def build_cresci_features(
    scan: CresciScan,
    max_tweets_per_user: int = 100,
    progress: Callable[[str], None] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return one row per account with the 31 features, ``label`` and ``subset``."""
    if use_cache and scan.features_cache and scan.features_cache.exists():
        if progress:
            progress(f"Loading cached features from {scan.features_cache.name}")
        return pd.read_csv(scan.features_cache)
    if not scan.available:
        raise FileNotFoundError(f"No {scan.kind} subsets found under {scan.root}")

    frames = []
    extractor = FeatureExtractor()
    for sub in scan.subsets:
        if progress:
            progress(f"Reading {sub.key}/users.csv")
        users = pd.read_csv(sub.users_file, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        users = normalise_columns(users)
        if "id" not in users.columns:
            raise ValueError(f"{sub.users_file} has no 'id' column")
        users["id"] = users["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        if sub.tweets_file is not None:
            agg = _aggregate_tweets(sub.tweets_file, max_tweets_per_user, progress=progress)
            merged = users.merge(agg, on="id", how="left")
        else:
            merged = users.copy()
        # users.csv may already contain 'url' (profile url) — keep it for profile_completeness
        feats = extractor.transform_frame(merged)
        feats["id"] = merged["id"].values
        feats["screen_name"] = merged["screen_name"].astype(str).values if "screen_name" in merged.columns else merged["id"].values
        feats["subset"] = sub.key
        feats["category"] = sub.category
        feats["label"] = sub.label
        feats["n_tweets"] = merged["n_tweets"].fillna(0).values if "n_tweets" in merged.columns else 0
        frames.append(feats)
        if progress:
            progress(f"{sub.key}: {len(feats):,} accounts featurised")
    df = pd.concat(frames, ignore_index=True)
    df = df[["id", "screen_name", "subset", "category", "label", "n_tweets"] + FEATURE_NAMES]
    cache = scan.root / "features.csv"
    try:
        df.to_csv(cache, index=False)
        scan.features_cache = cache
    except OSError as exc:  # pragma: no cover
        log.warning("Could not write feature cache %s: %s", cache, exc)
    return df


# --------------------------------------------------------------------------- #
# Generic CSV inspection
# --------------------------------------------------------------------------- #


def read_csv_safely(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", low_memory=False)


def inspect_dataframe(df: pd.DataFrame, preview_rows: int = 10) -> dict[str, Any]:
    """Summary used by the Datasets page."""
    norm = normalise_columns(df)
    label_col = detect_label_column(norm.columns)
    class_dist: dict[str, int] | None = None
    label_valid = 0
    if label_col is not None:
        labels = norm[label_col].map(coerce_label)
        label_valid = int(labels.notna().sum())
        class_dist = {"HUMAN": int((labels == 0).sum()), "BOT": int((labels == 1).sum()), "unknown": int(labels.isna().sum())}
    missing = {str(c): int(df[c].isna().sum()) for c in df.columns}
    columns = [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique(dropna=True)),
            "canonical": str(nc),
        }
        for c, nc in zip(df.columns, norm.columns)
    ]
    availability = FeatureExtractor.feature_availability(df.columns)
    preview = df.head(preview_rows).replace({np.nan: None}).to_dict(orient="records")
    warnings_out: list[str] = []
    if len(df) == 0:
        warnings_out.append("Dataset is empty")
    if availability["coverage"] < 0.5:
        warnings_out.append("Fewer than half of the paper's 31 features can be derived from these columns")
    if label_col and class_dist and (class_dist["HUMAN"] == 0 or class_dist["BOT"] == 0):
        warnings_out.append("Label column found but only one class present — evaluation/training requires both")
    numeric_summary = {}
    for f in FEATURE_NAMES:
        if f in norm.columns:
            col = pd.to_numeric(norm[f], errors="coerce")
            if col.notna().any():
                numeric_summary[f] = {
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "median": float(col.median()),
                }
    return json_safe(
        {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing": int(sum(missing.values())),
            "missing_by_column": missing,
            "columns": columns,
            "label_column": label_col,
            "label_valid_rows": label_valid,
            "class_distribution": class_dist,
            "feature_availability": availability,
            "numeric_summary": numeric_summary,
            "preview": preview,
            "warnings": warnings_out,
        }
    )


def labelled_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features, y) from a labelled CSV; rows with unknown labels are dropped."""
    norm = normalise_columns(df)
    label_col = detect_label_column(norm.columns)
    if label_col is None:
        raise ValueError("No label column found (expected one of: label, is_bot, bot, class, target, account_type)")
    y = norm[label_col].map(coerce_label)
    keep = y.notna()
    X = FeatureExtractor().transform_frame(norm[keep])
    return X, y[keep].astype(int).values
