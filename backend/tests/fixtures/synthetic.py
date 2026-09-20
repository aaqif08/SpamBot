"""Synthetic test fixtures — TEST DATA ONLY.

This module lives under ``tests/`` and is never imported by production code.
It generates labelled accounts with realistic feature ranges so the ML and API
test-suites can run without any real dataset. It must never be loaded into a
production database or shown in the client-facing UI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES, FeatureExtractor

TEST_SOURCE = "TEST_SYNTHETIC"

_HUMAN_TWEETS = [
    "Finally finished the marathon this morning. Legs are jelly but so happy!",
    "Anyone else think the new season of that show started slow? Second episode picked up though.",
    "Coffee, rain, and a good book. Sunday sorted.",
    "Congrats to my sister on her graduation! So proud of you.",
    "Our team lost again... we really need a new defender.",
    "Trying a new pasta recipe tonight, wish me luck",
    "Long day at work, but the sunset on the way home made up for it.",
    "Thanks @friend for the birthday wishes, you made my day :)",
    "Can't believe it's already October. Where did the year go?",
    "Reading about urban planning and now I have opinions about bike lanes.",
    "Kids built a blanket fort in the living room. I am not allowed in.",
    "Was the concert last night as loud for everyone else? Ears still ringing!",
]

_BOT_TWEETS = [
    "GET 5000 FOLLOWERS FAST!!! Click here http://bit.ly/x1 #followback #teamfollowback #follow4follow",
    "Make $500/day from home with this ONE trick http://tinyurl.com/abc #money #wfh #crypto",
    "RT @promo_deals: 90% OFF today only! http://bit.ly/deal #sale #deals #discount #shopping",
    "Best app of 2024!! Download now http://bit.ly/app #app #free #download #mobile #ios",
    "Buy cheap followers & likes http://bit.ly/f0ll #followers #likes #instagram #twitter",
    "New ebook FREE for 24h http://bit.ly/ebook #free #ebook #amazon #kindle #books",
    "@user1 @user2 @user3 @user4 check this out http://bit.ly/xyz #viral #trending",
    "AMAZING deal on phones http://bit.ly/ph0ne #phone #android #deal #cheap #offer",
]


def _lognormal(rng: np.random.RandomState, mean: float, sigma: float, size: int) -> np.ndarray:
    return np.exp(rng.normal(np.log(max(mean, 1e-6)), sigma, size)).round()


def generate_synthetic_accounts(n: int = 600, bot_ratio: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """Generate ``n`` synthetic accounts with all 31 features + label.

    Three bot archetypes are simulated to mirror the paper's dataset structure:
    *fake followers* (near-dormant, default profiles, follow many), *spambots*
    (hashtag/URL heavy, high posting volume) and *sophisticated bots* (human-like
    profiles with a few automated signals, producing genuine class overlap).
    Humans have moderate, noisy values.
    """
    rng = np.random.RandomState(seed)
    n_bot = int(round(n * bot_ratio))
    n_human = n - n_bot
    n_fake = int(n_bot * 0.35)
    n_spam = int(n_bot * 0.35)
    n_soph = n_bot - n_fake - n_spam
    rows: list[dict[str, Any]] = []

    def base_row(kind: str) -> dict[str, Any]:
        if kind == "human":
            statuses = _lognormal(rng, 1800, 1.1, 1)[0]
            followers = _lognormal(rng, 400, 1.3, 1)[0]
            friends = _lognormal(rng, 350, 1.0, 1)[0]
            listed = _lognormal(rng, 4, 1.2, 1)[0] * (rng.rand() < 0.7)
            favorites = _lognormal(rng, 900, 1.4, 1)[0]
            n_tweets = int(min(statuses, rng.randint(40, 200)))
            hashtag = rng.binomial(n_tweets, rng.uniform(0.05, 0.3))
            mentions = rng.binomial(n_tweets, rng.uniform(0.2, 0.7))
            urls = rng.binomial(n_tweets, rng.uniform(0.05, 0.3))
            retweets = _lognormal(rng, max(n_tweets * 0.5, 1), 1.0, 1)[0]
            replies = _lognormal(rng, max(n_tweets * 0.3, 1), 1.0, 1)[0]
            fav_recv = _lognormal(rng, max(n_tweets * 1.2, 1), 1.0, 1)[0]
            verified = rng.rand() < 0.04
            default_profile = rng.rand() < 0.25
            default_image = rng.rand() < 0.03
            geo = rng.rand() < 0.45
            tile = rng.rand() < 0.2
            desc_present = rng.rand() < 0.9
            uniq_words = _lognormal(rng, 600, 0.5, 1)[0]
            total_words = uniq_words * rng.uniform(1.6, 3.0)
            punct = total_words * rng.uniform(0.08, 0.2)
            chars = total_words * rng.uniform(5.0, 6.5)
            sent_len = rng.uniform(8, 20)
            polarity = rng.normal(0.12, 0.12)
            subjectivity = rng.uniform(0.35, 0.65)
            completeness = np.clip(rng.normal(0.75, 0.15), 0, 1)
        elif kind == "fake_follower":
            statuses = _lognormal(rng, 25, 1.4, 1)[0]
            followers = _lognormal(rng, 12, 1.2, 1)[0]
            friends = _lognormal(rng, 900, 0.9, 1)[0]
            listed = 0
            favorites = _lognormal(rng, 3, 1.5, 1)[0]
            n_tweets = int(max(1, min(statuses, rng.randint(1, 40))))
            hashtag = rng.binomial(n_tweets, rng.uniform(0.0, 0.2))
            mentions = rng.binomial(n_tweets, rng.uniform(0.0, 0.15))
            urls = rng.binomial(n_tweets, rng.uniform(0.0, 0.25))
            retweets = rng.binomial(n_tweets, 0.05)
            replies = rng.binomial(n_tweets, 0.02)
            fav_recv = rng.binomial(n_tweets, 0.05)
            verified = False
            default_profile = rng.rand() < 0.85
            default_image = rng.rand() < 0.55
            geo = rng.rand() < 0.05
            tile = rng.rand() < 0.1
            desc_present = rng.rand() < 0.35
            uniq_words = _lognormal(rng, 40, 0.8, 1)[0]
            total_words = uniq_words * rng.uniform(1.0, 1.4)
            punct = total_words * rng.uniform(0.02, 0.1)
            chars = total_words * rng.uniform(5.0, 6.0)
            sent_len = rng.uniform(4, 10)
            polarity = rng.normal(0.02, 0.05)
            subjectivity = rng.uniform(0.1, 0.4)
            completeness = np.clip(rng.normal(0.3, 0.15), 0, 1)
        else:  # spambot
            statuses = _lognormal(rng, 6000, 0.9, 1)[0]
            followers = _lognormal(rng, 150, 1.2, 1)[0]
            friends = _lognormal(rng, 1500, 0.8, 1)[0]
            listed = _lognormal(rng, 2, 1.0, 1)[0] * (rng.rand() < 0.4)
            favorites = _lognormal(rng, 30, 1.5, 1)[0]
            n_tweets = int(min(statuses, rng.randint(100, 200)))
            hashtag = int(n_tweets * rng.uniform(1.5, 4.5))
            mentions = int(n_tweets * rng.uniform(0.3, 2.5))
            urls = int(n_tweets * rng.uniform(0.7, 1.3))
            retweets = _lognormal(rng, max(n_tweets * 0.15, 1), 1.2, 1)[0]
            replies = rng.binomial(n_tweets, 0.03)
            fav_recv = rng.binomial(n_tweets, 0.08)
            verified = False
            default_profile = rng.rand() < 0.6
            default_image = rng.rand() < 0.2
            geo = rng.rand() < 0.1
            tile = rng.rand() < 0.35
            desc_present = rng.rand() < 0.75
            uniq_words = _lognormal(rng, 120, 0.6, 1)[0]
            total_words = uniq_words * rng.uniform(4.0, 9.0)
            punct = total_words * rng.uniform(0.15, 0.35)
            chars = total_words * rng.uniform(5.5, 7.5)
            sent_len = rng.uniform(6, 14)
            polarity = rng.normal(0.35, 0.15)
            subjectivity = rng.uniform(0.5, 0.85)
            completeness = np.clip(rng.normal(0.5, 0.15), 0, 1)

        n_tweets = max(int(n_tweets), 1)
        row = {
            "verified": int(verified),
            "friends_count": int(friends),
            "followers_count": int(followers),
            "listed_count": int(listed),
            "favorites_count": int(favorites),
            "hashtag_count": int(hashtag),
            "mentions_count": int(mentions),
            "retweet_count": int(retweets),
            "reply_count": int(replies),
            "url_count": int(urls),
            "statuses_count": int(statuses),
            "tweets_observed": n_tweets,
            "favorite_count_received": int(fav_recv),
            "unique_word_count": int(uniq_words),
            "unique_word_use": float(min(1.0, uniq_words / max(total_words, 1))),
            "punctuation_count": int(punct),
            "avg_sentence_length": float(sent_len),
            "punctuation_density": float(min(1.0, punct / max(chars, 1))),
            "profile_completeness": float(completeness),
            "description_binary": int(desc_present),
            "default_profile": int(default_profile),
            "default_profile_image": int(default_image),
            "geo_enabled": int(geo),
            "profile_background_tile": int(tile),
            "avg_polarity": float(np.clip(polarity, -1, 1)),
            "avg_subjectivity": float(np.clip(subjectivity, 0, 1)),
        }
        return row

    def sophisticated_row() -> dict[str, Any]:
        """Bots that mimic humans (paper: 'sophisticated social bots'): human-like
        profile with only a few automated signals. Produces genuine class overlap."""
        row = base_row("human")
        row["friends_count"] = int(row["friends_count"] * rng.uniform(1.8, 3.5))
        row["hashtag_count"] = int(row["hashtag_count"] * rng.uniform(1.5, 3.0))
        row["url_count"] = int(row["url_count"] * rng.uniform(1.3, 2.5))
        row["reply_count"] = int(row["reply_count"] * rng.uniform(0.2, 0.7))
        row["favorites_count"] = int(row["favorites_count"] * rng.uniform(0.1, 0.6))
        row["avg_polarity"] = float(np.clip(row["avg_polarity"] + rng.uniform(0.05, 0.25), -1, 1))
        row["unique_word_use"] = float(row["unique_word_use"] * rng.uniform(0.6, 0.95))
        row["geo_enabled"] = int(rng.rand() < 0.2)
        return row

    kinds = ["human"] * n_human + ["fake_follower"] * n_fake + ["spambot"] * n_spam + ["sophisticated_bot"] * n_soph
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        row = sophisticated_row() if kind == "sophisticated_bot" else base_row(kind)
        row["id"] = f"demo_{i + 1:05d}"
        row["screen_name"] = f"demo_user_{i + 1}"
        row["bot_type"] = kind
        row["label"] = 0 if kind == "human" else 1
        row["source"] = TEST_SOURCE
        rows.append(row)

    df = pd.DataFrame(rows)
    # derive engagement features with the real extractor so demo rows follow the
    # exact same formulas as production inputs
    feats = FeatureExtractor().transform_frame(df)
    for f in FEATURE_NAMES:
        df[f] = feats[f].values
    cols = ["id", "screen_name", "source", "bot_type", "label"] + FEATURE_NAMES + ["tweets_observed", "favorite_count_received"]
    return df[cols]


def synthetic_profiles() -> list[dict[str, Any]]:
    """Hand-written full-profile accounts (with tweets) for feature-extraction tests."""
    return [
        {
            "sample_id": "demo-human-1",
            "label_hint": "HUMAN-like",
                        "account": {
                "account_id": "demo_human_runner",
                "screen_name": "demo_human_runner",
                "name": "Alex Runner (demo)",
                "verified": False,
                "friends_count": 412,
                "followers_count": 538,
                "listed_count": 7,
                "favorites_count": 2310,
                "statuses_count": 4120,
                "description": "Weekend runner, coffee nerd, occasional photographer. Views are my own.",
                "location": "Manchester",
                "url": "https://example.org/alex",
                "default_profile": False,
                "default_profile_image": False,
                "geo_enabled": True,
                "profile_background_tile": False,
                "has_profile_banner": True,
                "favorite_count_received": 96,
                "tweets": [
                    {"text": t, "retweet_count": int(i % 3), "reply_count": int(i % 2), "favorite_count": 8}
                    for i, t in enumerate(_HUMAN_TWEETS)
                ],
            },
        },
        {
            "sample_id": "demo-spambot-1",
            "label_hint": "BOT-like (spambot)",
                        "account": {
                "account_id": "demo_promo_bot",
                "screen_name": "demo_promo_bot",
                "name": "Best Deals 24/7 (demo)",
                "verified": False,
                "friends_count": 2890,
                "followers_count": 143,
                "listed_count": 0,
                "favorites_count": 12,
                "statuses_count": 18750,
                "description": "FREE followers! Best deals! Follow back 100% #followback #deals",
                "location": "",
                "url": "",
                "default_profile": True,
                "default_profile_image": False,
                "geo_enabled": False,
                "profile_background_tile": True,
                "has_profile_banner": False,
                "favorite_count_received": 3,
                "tweets": [
                    {"text": t, "retweet_count": 0 if i % 4 else 2, "reply_count": 0, "favorite_count": 0}
                    for i, t in enumerate(_BOT_TWEETS * 2)
                ],
            },
        },
        {
            "sample_id": "demo-fake-follower-1",
            "label_hint": "BOT-like (fake follower)",
                        "account": {
                "account_id": "demo_fake_follower",
                "screen_name": "demo_fake_follower",
                "name": "jm93810 (demo)",
                "verified": False,
                "friends_count": 1206,
                "followers_count": 9,
                "listed_count": 0,
                "favorites_count": 1,
                "statuses_count": 14,
                "description": "",
                "location": "",
                "url": "",
                "default_profile": True,
                "default_profile_image": True,
                "geo_enabled": False,
                "profile_background_tile": False,
                "has_profile_banner": False,
                "favorite_count_received": 0,
                "tweets": [
                    {"text": "hello", "retweet_count": 0, "reply_count": 0, "favorite_count": 0},
                    {"text": "nice", "retweet_count": 0, "reply_count": 0, "favorite_count": 0},
                    {"text": "http://bit.ly/q1", "retweet_count": 0, "reply_count": 0, "favorite_count": 0},
                ],
            },
        },
    ]
