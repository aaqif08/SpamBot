"""Results **reported by the base paper** (Tables 5 and 6).

These numbers are reproduced verbatim from
Javed et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3551993, obtained by the
authors with 5-fold cross-validation on Cresci-15 / Cresci-17.

They are reference data only. The application always labels them
"Reported in base paper" and never mixes them with results measured by this
implementation.
"""

from __future__ import annotations

from typing import Any

PAPER_CITATION: dict[str, Any] = {
    "title": "Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning",
    "authors": [
        "Danish Javed",
        "Noor Zaman Jhanjhi",
        "Navid Ali Khan",
        "Sayan Kumar Ray",
        "Arafat Al-Dhaqm",
        "Victor R. Kebande",
    ],
    "venue": "IEEE Access, vol. 13, pp. 52246–52259",
    "year": 2025,
    "doi": "10.1109/ACCESS.2025.3551993",
    "license": "CC BY 4.0",
    "protocol": "5-fold cross-validation; stratified split; 31-feature compact set",
}

_ALGOS = [
    "random_forest",
    "svm",
    "decision_tree",
    "xgboost",
    "lightgbm",
    "logistic_regression",
    "extra_trees",
    "naive_bayes",
    "adaboost",
]

_C15 = [
    (0.990, 0.994, 0.990, 0.992, 0.999),
    (0.959, 0.972, 0.962, 0.967, 0.983),
    (0.976, 0.980, 0.982, 0.981, 0.974),
    (0.991, 0.994, 0.991, 0.993, 0.999),
    (0.991, 0.994, 0.992, 0.993, 0.999),
    (0.954, 0.973, 0.953, 0.963, 0.977),
    (0.987, 0.994, 0.986, 0.990, 0.999),
    (0.768, 0.739, 0.980, 0.842, 0.966),
    (0.986, 0.991, 0.988, 0.990, 0.998),
]

_C17 = [
    (0.988, 0.992, 0.992, 0.992, 0.998),
    (0.964, 0.981, 0.972, 0.977, 0.991),
    (0.984, 0.989, 0.989, 0.989, 0.978),
    (0.990, 0.994, 0.993, 0.993, 0.999),
    (0.990, 0.994, 0.993, 0.993, 0.999),
    (0.939, 0.964, 0.955, 0.981, 0.981),
    (0.987, 0.991, 0.992, 0.991, 0.998),
    (0.919, 0.992, 0.899, 0.944, 0.979),
    (0.983, 0.989, 0.990, 0.989, 0.997),
]


def _table(rows: list[tuple[float, float, float, float, float]]) -> list[dict[str, Any]]:
    return [
        {
            "algorithm": algo,
            "accuracy": r[0],
            "precision": r[1],
            "recall": r[2],
            "f1": r[3],
            "roc_auc": r[4],
        }
        for algo, r in zip(_ALGOS, rows)
    ]


PAPER_REPORTED_RESULTS: dict[str, Any] = {
    "source": "Reported in base paper (Tables 5 & 6) — NOT measured by this implementation",
    "citation": PAPER_CITATION,
    "datasets": {
        "cresci-15": {"table": "Table 5", "results": _table(_C15), "highlight": "LightGBM: accuracy 0.991, F1 0.993"},
        "cresci-17": {"table": "Table 6", "results": _table(_C17), "highlight": "XGBoost / LightGBM: accuracy 0.990, F1 0.993"},
    },
    "baselines": {
        "cresci-15": [
            {"cite": "[50] BotMoE", "accuracy": 0.985, "f1": 0.988},
            {"cite": "[51]", "accuracy": 0.978, "f1": 0.980},
            {"cite": "[52]", "accuracy": 0.988, "f1": 0.988},
            {"cite": "[53] SStackGNN", "accuracy": 0.977, "f1": 0.975},
            {"cite": "[54]", "accuracy": 0.972, "f1": 0.978},
            {"cite": "[55] BIC", "accuracy": 0.983, "f1": 0.987},
            {"cite": "Paper (LightGBM)", "accuracy": 0.991, "f1": 0.993},
        ],
        "cresci-17": [
            {"cite": "[15]", "accuracy": 0.980, "f1": 0.964},
            {"cite": "[34]", "accuracy": 0.985, "f1": 0.989},
            {"cite": "[56]", "accuracy": 0.982, "f1": 0.977},
            {"cite": "[57]", "accuracy": 0.956, "f1": 0.967},
            {"cite": "[58]", "accuracy": 0.967, "f1": 0.977},
            {"cite": "Paper (XGBoost)", "accuracy": 0.990, "f1": 0.993},
        ],
    },
}
