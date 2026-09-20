#!/usr/bin/env python
"""Generate docs/results.md from the model registry: measured hold-out and CV
metrics per dataset, plus cross-dataset evaluation of the active model.

Everything written by this script was measured by this implementation. The
paper's numbers are included in a clearly separated section for reference.

    python scripts/report_results.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.init()

import numpy as np  # noqa: E402

from app.db.database import SessionLocal  # noqa: E402
from app.services.dataset_service import DatasetService  # noqa: E402
from app.services.registry import get_predictor, get_registry  # noqa: E402
from ml.datasets import labelled_frame  # noqa: E402
from ml.evaluation import full_evaluation  # noqa: E402
from ml.paper_results import PAPER_REPORTED_RESULTS  # noqa: E402
from ml.train import MODEL_ZOO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDER = list(MODEL_ZOO)


def fmt(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    registry = get_registry()
    models = [m for m in registry.list() if not m.is_demo]
    active = registry.active()
    lines = [
        "# Experimental results — reproduced by this implementation",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/report_results.py` from `backend/models/registry.json`.",
        "",
        "All numbers in sections 1–2 were **measured by this implementation** on the real Cresci datasets "
        "(user-level mirror, see `backend/data/datasets/PROVENANCE.md`). They are **not** the paper's numbers; "
        "the paper's numbers appear only in section 3.",
        "",
        "Protocol: shuffle → stratified 75/25 hold-out → randomised hyperparameter search (F1, stratified 5-fold CV) "
        "on the training split → 5-fold CV of the tuned pipeline → hold-out evaluation. Tweet-derived features are "
        "constant without `tweets.csv` and are dropped automatically; the number of features actually used is shown per model.",
        "",
    ]

    by_dataset: dict[str, list] = {}
    for m in models:
        by_dataset.setdefault(m.dataset_name, []).append(m)

    lines.append("## 1. Hold-out and cross-validation results per dataset")
    for ds_name, ms in by_dataset.items():
        ms = sorted(ms, key=lambda m: ORDER.index(m.algorithm) if m.algorithm in ORDER else 99)
        lines += ["", f"### {ds_name}", ""]
        rows = []
        for m in ms:
            h, cv, sd = m.metrics["holdout"], m.metrics["cv"], m.metrics.get("cv_std", {})
            rows.append(
                {
                    "model": m.name + (" **(active)**" if active and m.id == active.id else ""),
                    "nf": m.n_features,
                    "acc": fmt(h["accuracy"]),
                    "prec": fmt(h["precision"]),
                    "rec": fmt(h["recall"]),
                    "f1": fmt(h["f1"]),
                    "auc": fmt(h["roc_auc"]),
                    "cvf1": f"{fmt(cv.get('f1'))} ± {fmt(sd.get('f1'))}",
                    "cvauc": f"{fmt(cv.get('roc_auc'))} ± {fmt(sd.get('roc_auc'))}",
                    "t": f"{m.training_seconds:.1f}s",
                }
            )
        lines.append(
            table(
                rows,
                [("model", "Classifier"), ("nf", "Features"), ("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"),
                 ("f1", "F1"), ("auc", "ROC-AUC"), ("cvf1", "CV F1 (mean ± std)"), ("cvauc", "CV AUC (mean ± std)"), ("t", "Train time")],
            )
        )

    # ---- cross-dataset ---------------------------------------------------- #
    lines += ["", "## 2. Cross-dataset generalisation (out-of-distribution)", ""]
    lines.append("For each Cresci dataset, the classifier with the highest hold-out F1 trained on that dataset is evaluated on the **other** dataset, which it never saw. The fake-follower accounts overlap between the two collections, so this is a partial rather than fully disjoint test; the row for the active model is in-sample when its training data includes the evaluation set.")
    lines.append("")
    predictor = get_predictor()
    db = SessionLocal()
    try:
        service = DatasetService(db)
        frames: dict[str, tuple] = {}
        for kind in ("cresci-15", "cresci-17"):
            status = service.cresci_status(kind)
            if status["imported"]:
                _, df = service.load_frame(status["dataset_id"])
                frames[kind] = labelled_frame(df)
    finally:
        db.close()

    def evaluate(model_id: str, kind: str) -> dict:
        X, y = frames[kind]
        proba = predictor.predict_vectors(X, model_id=model_id)
        ev = full_evaluation(y, (proba >= 0.5).astype(int), proba)
        cm = ev["confusion_matrix"]
        return {
            "ds": kind,
            "n": f"{len(y):,}",
            "acc": fmt(ev["metrics"]["accuracy"]),
            "prec": fmt(ev["metrics"]["precision"]),
            "rec": fmt(ev["metrics"]["recall"]),
            "f1": fmt(ev["metrics"]["f1"]),
            "auc": fmt(ev["metrics"]["roc_auc"]),
            "cm": f"TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']}",
        }

    rows = []
    for train_kind, eval_kind in (("cresci-17", "cresci-15"), ("cresci-15", "cresci-17")):
        candidates = [m for m in models if m.dataset_name.lower().startswith(train_kind)]
        if not candidates or eval_kind not in frames:
            continue
        best = max(candidates, key=lambda m: m.metrics["holdout"]["f1"])
        r = evaluate(best.id, eval_kind)
        r["model"] = f"{best.name} trained on {train_kind}"
        r["scope"] = "out-of-distribution"
        rows.append(r)
    if active is not None:
        for kind in frames:
            r = evaluate(active.id, kind)
            r["model"] = f"{active.name} (active, trained on {active.dataset_name})"
            r["scope"] = "in-sample" if "combined" in active.dataset_name.lower() or active.dataset_name.lower().startswith(kind) else "out-of-distribution"
            rows.append(r)
    if rows:
        lines.append(table(rows, [("model", "Model"), ("ds", "Evaluated on"), ("scope", "Scope"), ("n", "Accounts"), ("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("auc", "ROC-AUC"), ("cm", "Confusion")]))
    else:
        lines.append("_No Cresci datasets imported._")

    # ---- paper ---------------------------------------------------------------- #
    lines += ["", "## 3. Reported in base paper (reference only — NOT measured here)", ""]
    lines.append(f"{PAPER_REPORTED_RESULTS['citation']['title']}, {PAPER_REPORTED_RESULTS['citation']['venue']} ({PAPER_REPORTED_RESULTS['citation']['year']}), DOI {PAPER_REPORTED_RESULTS['citation']['doi']}. 5-fold CV with the full 31 features including tweet-derived ones.")
    for kind in ("cresci-15", "cresci-17"):
        block = PAPER_REPORTED_RESULTS["datasets"][kind]
        lines += ["", f"### {kind} ({block['table']})", ""]
        rows = [
            {"model": MODEL_ZOO[r["algorithm"]].display_name, "acc": fmt(r["accuracy"]), "prec": fmt(r["precision"]), "rec": fmt(r["recall"]), "f1": fmt(r["f1"]), "auc": fmt(r["roc_auc"])}
            for r in block["results"]
        ]
        lines.append(table(rows, [("model", "Classifier"), ("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("auc", "AUC")]))

    lines += [
        "",
        "## 4. Why our numbers differ from the paper",
        "",
        "- The public mirror contains only account profiles; the 11 tweet-derived features (hashtag/mention/URL/retweet/reply counts and their per-tweet averages, engagement) are unavailable and were dropped. Linguistic and sentiment features are computed from the profile description instead of tweets.",
        "- Hyperparameter search spaces and derived-feature formulas are ours (not published in the paper).",
        "- Paper results are 5-fold CV averages; our headline numbers are a stratified 25% hold-out (CV means are reported alongside).",
        "",
    ]
    out = ROOT / "docs" / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(models)} models, active={active.id if active else None})")


if __name__ == "__main__":
    main()
