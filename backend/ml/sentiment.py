"""Sentiment features (avg_polarity, avg_subjectivity).

The paper reports ``avg_polarity`` and ``avg_subjectivity`` — the exact output
names of TextBlob's PatternAnalyzer — so TextBlob is used. TextBlob's lexicon is
bundled with the package, so no corpus download is required for
``TextBlob(text).sentiment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .preprocessing import clean_for_sentiment


@dataclass(frozen=True)
class Sentiment:
    polarity: float
    subjectivity: float


def analyse(text: str) -> Sentiment:
    """Polarity in [-1, 1], subjectivity in [0, 1]. Empty text → neutral (0, 0)."""
    cleaned = clean_for_sentiment(text)
    if not cleaned:
        return Sentiment(0.0, 0.0)
    try:
        from textblob import TextBlob  # type: ignore

        s = TextBlob(cleaned).sentiment
        return Sentiment(float(s.polarity), float(s.subjectivity))
    except Exception:
        return Sentiment(0.0, 0.0)


def average_sentiment(texts: Iterable[str]) -> Sentiment:
    """Mean polarity/subjectivity over non-empty texts. No texts → (0, 0)."""
    pol_sum = 0.0
    sub_sum = 0.0
    n = 0
    for t in texts:
        if not t or not str(t).strip():
            continue
        s = analyse(str(t))
        pol_sum += s.polarity
        sub_sum += s.subjectivity
        n += 1
    if n == 0:
        return Sentiment(0.0, 0.0)
    return Sentiment(pol_sum / n, sub_sum / n)
