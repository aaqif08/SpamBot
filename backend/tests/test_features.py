"""Feature extraction and preprocessing tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml import preprocessing as pp
from ml.features import (
    AUXILIARY_FEATURES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FeatureExtractor,
    coerce_label,
    detect_label_column,
    normalise_columns,
)
from ml.sentiment import analyse, average_sentiment


def test_feature_set_matches_paper_table_4():
    assert len(FEATURE_NAMES) == 31
    assert {g: len(f) for g, f in FEATURE_GROUPS.items()} == {
        "user_profile": 5,
        "content": 6,
        "engagement": 7,
        "linguistic": 5,
        "profile_attributes": 6,
        "sentiment": 2,
    }
    assert len(set(FEATURE_NAMES)) == 31
    assert "avg_favorites" in AUXILIARY_FEATURES


def test_transform_full_account_derives_counts_from_tweets():
    fx = FeatureExtractor()
    account = {
        "verified": False,
        "friends_count": 100,
        "followers_count": 25,
        "listed_count": 1,
        "favorites_count": 10,
        "statuses_count": 400,
        "description": "Deals every day #promo http://x.co",
        "default_profile": True,
        "tweets": [
            {"text": "Buy now http://bit.ly/a #deal #sale @shop", "retweet_count": 2, "reply_count": 0, "favorite_count": 1},
            {"text": "Another one http://bit.ly/b #deal", "retweet_count": 0, "reply_count": 1, "favorite_count": 0},
            "plain tweet without anything",
        ],
    }
    r = fx.transform(account)
    f = r.features
    assert set(f) == set(FEATURE_NAMES)
    assert f["hashtag_count"] == 3
    assert f["url_count"] == 2
    assert f["mentions_count"] == 1
    assert f["retweet_count"] == 2
    assert f["reply_count"] == 1
    assert f["statuses_count"] == 400
    assert math.isclose(f["ffratio"], 25 / 101)
    assert math.isclose(f["avg_hashtag"], 3 / 3)
    assert math.isclose(f["avg_url"], 2 / 3)
    assert math.isclose(f["avg_user_engagement"], (2 + 1 + 1) / 3)
    assert f["description_binary"] == 1.0
    assert f["default_profile"] == 1.0
    assert 0 < f["unique_word_use"] <= 1
    assert f["unique_word_count"] > 0
    assert f["punctuation_count"] > 0
    assert 0 <= f["punctuation_density"] < 1
    assert -1 <= f["avg_polarity"] <= 1
    assert 0 <= f["avg_subjectivity"] <= 1
    assert r.auxiliary["n_tweets"] == 3
    assert all(np.isfinite(v) for v in f.values())


def test_explicit_counts_override_tweet_derivation():
    fx = FeatureExtractor()
    r = fx.transform({"statuses_count": 10, "hashtag_count": 50, "tweets": ["#a #b"]})
    assert r.features["hashtag_count"] == 50


def test_missing_and_invalid_values_are_handled():
    fx = FeatureExtractor()
    r = fx.transform({"friends_count": "nan", "followers_count": None, "statuses_count": -5, "description": None, "tweets": None})
    f = r.features
    assert f["friends_count"] == 0
    assert f["followers_count"] == 0
    assert f["statuses_count"] == 0
    assert f["description_binary"] == 0
    assert f["avg_polarity"] == 0 and f["avg_subjectivity"] == 0
    assert all(np.isfinite(v) for v in f.values())
    assert r.auxiliary["description_imputed"] == 1.0
    assert r.auxiliary["description_length"] == 0


def test_profile_completeness_counts_filled_fields():
    fx = FeatureExtractor()
    full = fx.transform({"name": "A", "description": "b", "location": "c", "url": "d", "default_profile_image": False, "has_profile_banner": True})
    empty = fx.transform({"default_profile_image": True})
    assert full.features["profile_completeness"] == 1.0
    assert empty.features["profile_completeness"] == 0.0


def test_transform_frame_with_cresci_aliases():
    df = pd.DataFrame(
        [
            {"id": 1, "favourites_count": 3, "friends_count": 10, "followers_count": 5, "statuses_count": 20, "num_hashtags": 4, "num_urls": 2, "num_mentions": 1, "text": "hello #x"},
            {"id": 2, "favourites_count": 0, "friends_count": 0, "followers_count": 0, "statuses_count": 0, "num_hashtags": 0, "num_urls": 0, "num_mentions": 0, "text": ""},
        ]
    )
    out = FeatureExtractor().transform_frame(df)
    assert list(out.columns) == FEATURE_NAMES
    assert out.loc[0, "favorites_count"] == 3
    assert out.loc[0, "hashtag_count"] == 4
    assert out.loc[1, "ffratio"] == 0


def test_feature_availability_report():
    rep = FeatureExtractor.feature_availability(["followers_count", "friends_count", "text", "label"])
    assert "followers_count" in rep["direct"]
    assert "ffratio" in rep["derivable"]
    assert "verified" in rep["missing"]
    assert 0 < rep["coverage"] < 1


def test_label_detection_and_coercion():
    assert detect_label_column(["id", "Is_Bot"]) == "Is_Bot"
    assert detect_label_column(["id", "score"]) is None
    assert coerce_label("bot") == 1 and coerce_label("human") == 0
    assert coerce_label(1.0) == 1 and coerce_label("0") == 0
    assert coerce_label("weird") is None and coerce_label(float("nan")) is None


def test_normalise_columns_drops_duplicates():
    df = pd.DataFrame(columns=["Favourites_Count", "favorites_count"])
    assert list(normalise_columns(df).columns) == ["favorites_count"]


# ---- preprocessing ------------------------------------------------------------- #


def test_sentiment_path_cleans_text():
    cleaned = pp.clean_for_sentiment("Check THIS out!!! http://bit.ly/x @user #Great 😀 the and of")
    assert "http" not in cleaned and "@user" not in cleaned
    assert "great" in cleaned
    assert "grinning" in cleaned  # emoji converted to text
    assert " the " not in f" {cleaned} "  # stop word removed


def test_feature_path_counts_keep_urls_and_punctuation():
    text = "Wow!!! visit http://a.co and www.b.com @x @y #z"
    assert pp.count_urls(text) == 2
    assert pp.count_mentions(text) == 2
    assert pp.count_hashtags(text) == 1
    assert pp.count_punctuation(text) >= 3
    assert pp.is_retweet_text("RT @a: hi") and not pp.is_retweet_text("hi RT")


def test_description_imputation_rule():
    assert pp.impute_description(None) == "missing"
    assert pp.impute_description("  ") == "missing"
    assert pp.impute_description("hello") == "hello"
    assert pp.description_length(None) == 0 and pp.description_length("abc") == 3


def test_text_stats():
    st = pp.text_stats(["One two three. Four five!", "one two"])
    assert st.total_words == 7
    assert st.unique_words == 5
    assert st.sentence_count == 3
    assert 0 < st.unique_word_use <= 1
    assert st.avg_sentence_length == pytest.approx(7 / 3)


def test_sentiment_values_in_range():
    s = analyse("I absolutely love this wonderful product")
    assert s.polarity > 0 and 0 <= s.subjectivity <= 1
    assert average_sentiment([]) == analyse("")
    avg = average_sentiment(["great", "terrible", ""])
    assert -1 <= avg.polarity <= 1
