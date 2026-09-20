"""Feature engineering — the single source of truth for the paper's 31 features.

Table 4 of the base paper defines six feature groups. ``FEATURE_GROUPS`` below
is the only place where the canonical feature list lives; training, prediction,
explanation and the API all import it from here.

Input forms accepted by :class:`FeatureExtractor`:

1. **Account dict** — profile counts/flags, ``description`` and an optional
   list of ``tweets`` (strings or dicts). Content counts are derived from the
   tweets when not supplied explicitly.
2. **Pre-aggregated row** (CSV) — columns already containing counts. Column
   aliases from the Cresci ``users.csv`` schema are recognised.

Derived-feature formulas are not spelled out in the paper; the definitions used
here are documented in ``docs/methodology.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import FEATURE_VERSION
from .preprocessing import (
    count_hashtags,
    count_mentions,
    count_urls,
    description_length,
    impute_description,
    is_retweet_text,
    text_stats,
)
from .sentiment import average_sentiment
from .utils import safe_div, to_bool, to_float, to_text

# --------------------------------------------------------------------------- #
# Canonical feature definition (paper Table 4)
# --------------------------------------------------------------------------- #

FEATURE_GROUPS: dict[str, list[str]] = {
    "user_profile": [
        "verified",
        "friends_count",
        "followers_count",
        "listed_count",
        "favorites_count",
    ],
    "content": [
        "hashtag_count",
        "mentions_count",
        "retweet_count",
        "reply_count",
        "url_count",
        "statuses_count",
    ],
    "engagement": [
        "ffratio",
        "avg_hashtag",
        "avg_retweets",
        "avg_replies",
        "avg_mentions",
        "avg_url",
        "avg_user_engagement",
    ],
    "linguistic": [
        "unique_word_count",
        "unique_word_use",
        "punctuation_count",
        "avg_sentence_length",
        "punctuation_density",
    ],
    "profile_attributes": [
        "profile_completeness",
        "description_binary",
        "default_profile",
        "default_profile_image",
        "geo_enabled",
        "profile_background_tile",
    ],
    "sentiment": [
        "avg_polarity",
        "avg_subjectivity",
    ],
}

FEATURE_NAMES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]
assert len(FEATURE_NAMES) == 31, "Paper feature set must contain exactly 31 features"

#: Auxiliary features that are computed and stored but not part of the canonical
#: 31-vector. ``avg_favorites`` appears in the paper's SHAP/LIME figures but not
#: in Table 4; ``description_length`` is mentioned in §III-B.
AUXILIARY_FEATURES: list[str] = ["avg_favorites", "description_length", "n_tweets"]

FEATURE_GROUP_OF: dict[str, str] = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

#: Human-readable labels and short explanations for the UI.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "verified": "Account carries the platform's verified badge",
    "friends_count": "Number of accounts this account follows",
    "followers_count": "Number of accounts following this account",
    "listed_count": "Number of public lists the account appears in",
    "favorites_count": "Number of tweets the account has liked",
    "hashtag_count": "Total hashtags across the observed tweets",
    "mentions_count": "Total @-mentions across the observed tweets",
    "retweet_count": "Total retweets received by the observed tweets",
    "reply_count": "Total replies received by the observed tweets",
    "url_count": "Total URLs across the observed tweets",
    "statuses_count": "Total number of tweets posted (profile status count)",
    "ffratio": "Followers / (friends + 1)",
    "avg_hashtag": "Hashtags per observed tweet",
    "avg_retweets": "Retweets received per observed tweet",
    "avg_replies": "Replies received per observed tweet",
    "avg_mentions": "Mentions per observed tweet",
    "avg_url": "URLs per observed tweet",
    "avg_user_engagement": "(retweets + replies + likes received) per observed tweet",
    "unique_word_count": "Distinct words used across observed tweets",
    "unique_word_use": "Distinct words / total words (lexical diversity)",
    "punctuation_count": "Punctuation characters across observed tweets",
    "avg_sentence_length": "Average words per sentence",
    "punctuation_density": "Punctuation characters / total characters",
    "profile_completeness": "Share of profile fields filled (name, description, location, url, picture, banner)",
    "description_binary": "Description present (1) or missing (0)",
    "default_profile": "Profile theme left at the default",
    "default_profile_image": "Profile picture left at the default",
    "geo_enabled": "Geo-tagging enabled on the account",
    "profile_background_tile": "Background image set to tile",
    "avg_polarity": "Mean sentiment polarity of cleaned tweet text (−1…1)",
    "avg_subjectivity": "Mean sentiment subjectivity of cleaned tweet text (0…1)",
}

#: Feature names that are boolean-like (used for UI formatting).
BINARY_FEATURES: frozenset[str] = frozenset(
    {
        "verified",
        "description_binary",
        "default_profile",
        "default_profile_image",
        "geo_enabled",
        "profile_background_tile",
    }
)

# --------------------------------------------------------------------------- #
# Column aliases (Cresci users.csv / common variants → canonical names)
# --------------------------------------------------------------------------- #

COLUMN_ALIASES: dict[str, str] = {
    "favourites_count": "favorites_count",
    "favourite_count": "favorites_count",
    "favorite_count": "favorites_count",
    "likes_count": "favorites_count",
    "status_count": "statuses_count",
    "tweet_count": "statuses_count",
    "following_count": "friends_count",
    "friend_count": "friends_count",
    "follower_count": "followers_count",
    "num_hashtags": "hashtag_count",
    "hashtags_count": "hashtag_count",
    "num_mentions": "mentions_count",
    "mention_count": "mentions_count",
    "num_urls": "url_count",
    "urls_count": "url_count",
    "retweets_count": "retweet_count",
    "replies_count": "reply_count",
    "follower_following_ratio": "ffratio",
    "ff_ratio": "ffratio",
    "unique_word_usage": "unique_word_use",
    "avg_hashtags": "avg_hashtag",
    "avg_urls": "avg_url",
    "avg_reply": "avg_replies",
    "avg_retweet": "avg_retweets",
    "avg_mention": "avg_mentions",
    "avg_engagement": "avg_user_engagement",
    "has_description": "description_binary",
    "polarity": "avg_polarity",
    "subjectivity": "avg_subjectivity",
    "screenname": "screen_name",
    "user_id": "id",
    "account_id": "id",
}

LABEL_ALIASES: tuple[str, ...] = ("label", "is_bot", "bot", "class", "target", "y", "account_type")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip and alias-map DataFrame columns to canonical names."""
    renamed: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[col] = COLUMN_ALIASES.get(key, key)
    out = df.rename(columns=renamed)
    # drop duplicate columns produced by aliasing, keeping the first
    return out.loc[:, ~out.columns.duplicated()]


def detect_label_column(columns: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in columns}
    for candidate in LABEL_ALIASES:
        if candidate in cols:
            return cols[candidate]
    return None


def coerce_label(value: Any) -> int | None:
    """Map many label spellings to 1 (bot) / 0 (human); ``None`` if unknown."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(f):
            return None
        return 1 if f >= 0.5 else 0
    s = str(value).strip().lower()
    if s in ("1", "bot", "bots", "spambot", "fake", "fake_follower", "fake followers", "true", "yes", "automated"):
        return 1
    if s in ("0", "human", "humans", "genuine", "legit", "legitimate", "real", "false", "no"):
        return 0
    return None


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


@dataclass
class TweetRecord:
    text: str = ""
    retweet_count: float = 0.0
    reply_count: float = 0.0
    favorite_count: float = 0.0
    num_hashtags: float | None = None
    num_mentions: float | None = None
    num_urls: float | None = None

    @classmethod
    def from_any(cls, item: Any) -> "TweetRecord":
        if isinstance(item, TweetRecord):
            return item
        if isinstance(item, str):
            return cls(text=item)
        if isinstance(item, Mapping):
            return cls(
                text=to_text(item.get("text", item.get("tweet", item.get("content", "")))),
                retweet_count=to_float(item.get("retweet_count")),
                reply_count=to_float(item.get("reply_count")),
                favorite_count=to_float(item.get("favorite_count", item.get("favourite_count", item.get("like_count")))),
                num_hashtags=_opt_float(item.get("num_hashtags", item.get("hashtag_count"))),
                num_mentions=_opt_float(item.get("num_mentions", item.get("mentions_count"))),
                num_urls=_opt_float(item.get("num_urls", item.get("url_count"))),
            )
        return cls(text=to_text(item))


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return to_float(value)


@dataclass
class FeatureResult:
    """Output of :meth:`FeatureExtractor.transform`."""

    features: dict[str, float]
    auxiliary: dict[str, float] = field(default_factory=dict)

    def vector(self, names: Sequence[str] | None = None) -> np.ndarray:
        names = list(names) if names is not None else FEATURE_NAMES
        return np.array([float(self.features.get(n, 0.0)) for n in names], dtype=float)


class FeatureExtractor:
    """Builds the 31-feature vector from account data.

    Parameters
    ----------
    feature_names:
        Ordered list of features to output (defaults to the full paper set;
        a SHAP-selected subset can be passed after feature selection).
    """

    version = FEATURE_VERSION

    def __init__(self, feature_names: Sequence[str] | None = None) -> None:
        self.feature_names: list[str] = list(feature_names) if feature_names else list(FEATURE_NAMES)
        unknown = [f for f in self.feature_names if f not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"Unknown features requested: {unknown}")

    # ---- group extractors -------------------------------------------------- #

    @staticmethod
    def extract_user_features(account: Mapping[str, Any]) -> dict[str, float]:
        return {
            "verified": float(to_bool(account.get("verified"))),
            "friends_count": max(0.0, to_float(account.get("friends_count"))),
            "followers_count": max(0.0, to_float(account.get("followers_count"))),
            "listed_count": max(0.0, to_float(account.get("listed_count"))),
            "favorites_count": max(0.0, to_float(account.get("favorites_count"))),
        }

    @staticmethod
    def extract_content_features(account: Mapping[str, Any], tweets: Sequence[TweetRecord]) -> dict[str, float]:
        """Content counts. Explicit account-level counts win; otherwise derived from tweets."""

        def explicit(name: str) -> float | None:
            return _opt_float(account.get(name))

        hashtags = explicit("hashtag_count")
        mentions = explicit("mentions_count")
        urls = explicit("url_count")
        retweets = explicit("retweet_count")
        replies = explicit("reply_count")

        if tweets:
            if hashtags is None:
                hashtags = float(sum(t.num_hashtags if t.num_hashtags is not None else count_hashtags(t.text) for t in tweets))
            if mentions is None:
                mentions = float(sum(t.num_mentions if t.num_mentions is not None else count_mentions(t.text) for t in tweets))
            if urls is None:
                urls = float(sum(t.num_urls if t.num_urls is not None else count_urls(t.text) for t in tweets))
            if retweets is None:
                retweets = float(sum(t.retweet_count for t in tweets))
            if replies is None:
                replies = float(sum(t.reply_count for t in tweets))

        statuses = to_float(account.get("statuses_count"))
        if statuses <= 0 and tweets:
            statuses = float(len(tweets))

        return {
            "hashtag_count": max(0.0, hashtags or 0.0),
            "mentions_count": max(0.0, mentions or 0.0),
            "retweet_count": max(0.0, retweets or 0.0),
            "reply_count": max(0.0, replies or 0.0),
            "url_count": max(0.0, urls or 0.0),
            "statuses_count": max(0.0, statuses),
        }

    @staticmethod
    def observed_tweet_count(account: Mapping[str, Any], tweets: Sequence[TweetRecord], statuses: float) -> float:
        """Denominator for per-tweet averages.

        Number of tweets actually observed if provided, else an explicit
        ``tweets_observed`` value, else the profile status count.
        """
        if tweets:
            return float(len(tweets))
        observed = _opt_float(account.get("tweets_observed"))
        if observed is not None and observed > 0:
            return observed
        if statuses > 0:
            return statuses
        return 0.0

    @staticmethod
    def extract_engagement_features(
        user: Mapping[str, float],
        content: Mapping[str, float],
        n_tweets: float,
        favorites_received: float,
    ) -> dict[str, float]:
        return {
            "ffratio": safe_div(user["followers_count"], user["friends_count"] + 1.0),
            "avg_hashtag": safe_div(content["hashtag_count"], n_tweets),
            "avg_retweets": safe_div(content["retweet_count"], n_tweets),
            "avg_replies": safe_div(content["reply_count"], n_tweets),
            "avg_mentions": safe_div(content["mentions_count"], n_tweets),
            "avg_url": safe_div(content["url_count"], n_tweets),
            "avg_user_engagement": safe_div(
                content["retweet_count"] + content["reply_count"] + favorites_received, n_tweets
            ),
        }

    @staticmethod
    def extract_linguistic_features(texts: Sequence[str]) -> dict[str, float]:
        stats = text_stats(texts)
        return {
            "unique_word_count": float(stats.unique_words),
            "unique_word_use": float(stats.unique_word_use),
            "punctuation_count": float(stats.punctuation_count),
            "avg_sentence_length": float(stats.avg_sentence_length),
            "punctuation_density": float(stats.punctuation_density),
        }

    @staticmethod
    def extract_profile_features(account: Mapping[str, Any]) -> dict[str, float]:
        description = to_text(account.get("description")).strip()
        default_image = to_bool(account.get("default_profile_image"))
        has_image = bool(to_text(account.get("profile_image_url")).strip()) and not default_image
        if "profile_image_url" not in account:
            has_image = not default_image
        has_banner = to_bool(account.get("has_profile_banner")) or bool(to_text(account.get("profile_banner_url")).strip())
        components = [
            bool(to_text(account.get("name")).strip()),
            bool(description),
            bool(to_text(account.get("location")).strip()),
            bool(to_text(account.get("url")).strip()),
            has_image,
            has_banner,
        ]
        return {
            "profile_completeness": sum(components) / len(components),
            "description_binary": 1.0 if description else 0.0,
            "default_profile": float(to_bool(account.get("default_profile"))),
            "default_profile_image": float(default_image),
            "geo_enabled": float(to_bool(account.get("geo_enabled"))),
            "profile_background_tile": float(to_bool(account.get("profile_background_tile"))),
        }

    @staticmethod
    def extract_sentiment_features(texts: Sequence[str]) -> dict[str, float]:
        s = average_sentiment(texts)
        return {"avg_polarity": float(s.polarity), "avg_subjectivity": float(s.subjectivity)}

    # ---- public API -------------------------------------------------------- #

    def transform(self, account: Mapping[str, Any]) -> FeatureResult:
        """Compute all features for one account (dict form)."""
        tweets = [TweetRecord.from_any(t) for t in (account.get("tweets") or [])]
        tweet_texts = [t.text for t in tweets if t.text and t.text.strip()]

        user = self.extract_user_features(account)
        content = self.extract_content_features(account, tweets)
        n_tweets = self.observed_tweet_count(account, tweets, content["statuses_count"])

        favorites_received = _opt_float(account.get("favorite_count_received"))
        if favorites_received is None:
            favorites_received = float(sum(t.favorite_count for t in tweets)) if tweets else 0.0

        engagement = self.extract_engagement_features(user, content, n_tweets, favorites_received)

        # Linguistic + sentiment operate on tweet text; fall back to the
        # description when no tweets are available so the features are not
        # trivially zero for profile-only inputs.
        desc_raw = to_text(account.get("description")).strip()
        ling_source = tweet_texts if tweet_texts else ([desc_raw] if desc_raw else [])
        linguistic = self.extract_linguistic_features(ling_source)

        sentiment_source = list(tweet_texts)
        if desc_raw:
            sentiment_source.append(desc_raw)
        sentiment = self.extract_sentiment_features(sentiment_source)

        profile = self.extract_profile_features(account)

        # Pre-aggregated rows may already carry derived/linguistic/sentiment
        # columns (e.g. a CSV exported by this system). Explicit values win.
        computed: dict[str, float] = {**user, **content, **engagement, **linguistic, **profile, **sentiment}
        for name in FEATURE_NAMES:
            if name in (*user, *content):
                continue
            explicit = _opt_float(account.get(name))
            if explicit is not None and not (name in ("avg_polarity", "avg_subjectivity") and sentiment_source):
                computed[name] = explicit

        # Paper §III-B: description imputed with "missing"; length stays 0.
        aux = {
            "avg_favorites": safe_div(favorites_received, n_tweets),
            "description_length": float(description_length(account.get("description"))),
            "n_tweets": float(n_tweets),
            "description_imputed": 1.0 if impute_description(account.get("description")) == "missing" else 0.0,
        }
        features = {name: float(computed.get(name, 0.0)) for name in self.feature_names}
        return FeatureResult(features=features, auxiliary=aux)

    def transform_many(self, accounts: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
        rows = [self.transform(a).features for a in accounts]
        return pd.DataFrame(rows, columns=self.feature_names)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a pre-aggregated DataFrame (one account per row)."""
        norm = normalise_columns(df)
        records = norm.to_dict(orient="records")
        out = self.transform_many(records)
        out.index = df.index
        return out

    # ---- schema helpers ---------------------------------------------------- #

    @staticmethod
    def feature_availability(columns: Iterable[str]) -> dict[str, Any]:
        """Report which canonical features a CSV can provide directly or derive."""
        cols = set(normalise_columns(pd.DataFrame(columns=list(columns))).columns)
        direct = [f for f in FEATURE_NAMES if f in cols]
        derivable: list[str] = []
        has_profile_counts = {"followers_count", "friends_count"} <= cols
        has_tweet_counts = {"hashtag_count", "mentions_count", "url_count"} & cols
        has_text = bool({"text", "tweets", "description"} & cols)
        for f in FEATURE_NAMES:
            if f in direct:
                continue
            group = FEATURE_GROUP_OF[f]
            if group == "engagement" and (has_profile_counts or has_tweet_counts):
                derivable.append(f)
            elif group in ("linguistic", "sentiment") and has_text:
                derivable.append(f)
            elif f in ("description_binary", "profile_completeness") and has_text:
                derivable.append(f)
        missing = [f for f in FEATURE_NAMES if f not in direct and f not in derivable]
        return {
            "direct": direct,
            "derivable": derivable,
            "missing": missing,
            "coverage": (len(direct) + len(derivable)) / len(FEATURE_NAMES),
        }


def features_by_group(values: Mapping[str, float]) -> dict[str, dict[str, float]]:
    return {g: {f: float(values.get(f, 0.0)) for f in fs if f in values} for g, fs in FEATURE_GROUPS.items()}
