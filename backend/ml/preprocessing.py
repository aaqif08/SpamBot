"""Text preprocessing following the base paper (§III-B).

The paper describes two text paths:

* **Feature path** – raw text is kept, because URL / mention / hashtag /
  punctuation information are themselves model features.
* **Sentiment path** – emojis are converted to text, URLs/mentions are removed,
  special characters, punctuation and whitespace are standardised, and stop
  words are removed before sentiment analysis.

Null ``description`` values are imputed with the literal ``"missing"`` while
``description_length`` stays 0 (paper §III-B).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .utils import to_text

MISSING_DESCRIPTION_TOKEN = "missing"

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9.-]+\.(com|net|org|ly|co|io|me)\b\S*)", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<![\w])@\w{1,30}")
HASHTAG_RE = re.compile(r"(?<![\w])#\w+")
RT_RE = re.compile(r"^\s*RT\s+@\w+", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|\n+")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")

PUNCTUATION_SET = set(string.punctuation)


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    """English stop words. NLTK corpus if present locally, sklearn list otherwise.

    No network download is attempted at runtime so that the API never blocks
    on first request.
    """
    try:
        from nltk.corpus import stopwords  # type: ignore

        words = set(stopwords.words("english"))
        if words:
            return frozenset(words)
    except Exception:  # LookupError when corpus missing, ImportError, etc.
        pass
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


def demojize(text: str) -> str:
    """Convert emojis to their textual name so they can influence sentiment."""
    try:
        import emoji  # type: ignore

        out = emoji.demojize(text, delimiters=(" ", " "))
        return out.replace("_", " ")
    except Exception:
        return text


def impute_description(description: object) -> str:
    """Paper: null descriptions become the literal token ``missing``."""
    text = to_text(description).strip()
    return text if text else MISSING_DESCRIPTION_TOKEN


def description_length(description: object) -> int:
    """Paper: description_length stays 0 for null descriptions."""
    text = to_text(description).strip()
    return len(text)


def clean_for_sentiment(text: object) -> str:
    """Sentiment path: emoji→text, drop URLs/mentions, normalise, drop stop words."""
    raw = to_text(text)
    if not raw:
        return ""
    t = demojize(raw)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    t = t.lower()
    t = NON_ALPHA_RE.sub(" ", t)
    tokens = [tok for tok in WHITESPACE_RE.split(t) if tok and tok not in stop_words()]
    return " ".join(tokens)


def tokenize_words(text: object) -> list[str]:
    """Feature path tokenisation (raw text, lower-cased alphanumerics, URLs removed)."""
    raw = to_text(text)
    if not raw:
        return []
    t = URL_RE.sub(" ", raw)
    return [w.lower() for w in WORD_RE.findall(t)]


def count_punctuation(text: object) -> int:
    raw = to_text(text)
    return sum(1 for ch in raw if ch in PUNCTUATION_SET)


def count_urls(text: object) -> int:
    return len(URL_RE.findall(to_text(text)))


def count_mentions(text: object) -> int:
    return len(MENTION_RE.findall(to_text(text)))


def count_hashtags(text: object) -> int:
    return len(HASHTAG_RE.findall(to_text(text)))


def is_retweet_text(text: object) -> bool:
    return bool(RT_RE.match(to_text(text)))


def split_sentences(text: object) -> list[str]:
    raw = to_text(text)
    if not raw.strip():
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(raw)]
    return [p for p in parts if p]


@dataclass
class TextStats:
    """Linguistic statistics for a bag of texts (feature path)."""

    total_words: int = 0
    unique_words: int = 0
    punctuation_count: int = 0
    total_chars: int = 0
    sentence_count: int = 0

    @property
    def unique_word_use(self) -> float:
        return self.unique_words / self.total_words if self.total_words else 0.0

    @property
    def avg_sentence_length(self) -> float:
        return self.total_words / self.sentence_count if self.sentence_count else 0.0

    @property
    def punctuation_density(self) -> float:
        return self.punctuation_count / self.total_chars if self.total_chars else 0.0


def text_stats(texts: Iterable[object]) -> TextStats:
    vocab: set[str] = set()
    stats = TextStats()
    for raw in texts:
        t = to_text(raw)
        if not t.strip():
            continue
        words = tokenize_words(t)
        stats.total_words += len(words)
        vocab.update(words)
        stats.punctuation_count += count_punctuation(t)
        stats.total_chars += len(t)
        sentences = split_sentences(t)
        stats.sentence_count += len(sentences) if sentences else 1
    stats.unique_words = len(vocab)
    return stats
