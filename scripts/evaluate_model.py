#!/usr/bin/env python
"""Evaluate a trained model on a labelled dataset and print/save the metrics.

Examples
--------
    python scripts/evaluate_model.py                       # active model, its own stored hold-out metrics
    python scripts/evaluate_model.py --dataset demo        # active model on the demo dataset
    python scripts/evaluate_model.py --model <id> --csv labelled.csv --out eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.init()

import numpy as np  # noqa: E402

from app.db.database import SessionLocal  # noqa: E402
from app.services.dataset_service import DatasetService  # noqa: E402
from app.services.registry import get_predictor, get_registry  # noqa: E402
from ml.datasets import labelled_frame, read_csv_safely  # noqa: E402
from ml.evaluation import full_evaluation  # noqa: E402
from ml.utils import json_safe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Model id (default: active model)")
    p.add_argument("--dataset", default=None, help="demo | cresci-15 | cresci-17 | <dataset id>")
    p.add_argument("--csv", default=None, help="Labelled CSV path")
    p.add_argument("--out", default=None, help="Write full evaluation JSON here")
    args = p.parse_args()

    registry = get_registry()
    predictor = get_predictor()
    try:
        model = predictor.get(args.model)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"No model available: {exc}")
    print(f"Model: {model.entry.name} [{model.entry.id}] trained on {model.entry.dataset_name} (demo={model.entry.is_demo})")

    if args.csv is None and args.dataset is None:
        print("\nStored hold-out evaluation (from training):")
        print(json.dumps(model.metrics["holdout"]["metrics"], indent=2))
        print("Cross-validation summary:")
        print(json.dumps(model.metrics["cross_validation"]["summary"], indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(json_safe(model.metrics), indent=2), encoding="utf-8")
        return

    if args.csv:
        df = read_csv_safely(Path(args.csv))
        name = Path(args.csv).name
        is_demo = False
    else:
        db = SessionLocal()
        try:
            service = DatasetService(db)
            if args.dataset == "demo":
                info = service.create_demo()
                ds_id = info["id"]
            elif args.dataset in ("cresci-15", "cresci-17"):
                status = service.cresci_status(args.dataset)
                if not status["imported"]:
                    sys.exit(f"{args.dataset} is not imported. Run scripts/train_model.py --dataset {args.dataset} first.")
                ds_id = status["dataset_id"]
            else:
                ds_id = args.dataset
            row, df = service.load_frame(ds_id)
            name, is_demo = row.name, bool(row.is_demo)
        finally:
            db.close()

    X, y = labelled_frame(df)
    proba = predictor.predict_vectors(X, model_id=model.entry.id)
    ev = full_evaluation(y, (proba >= 0.5).astype(int), proba)
    print(f"\nEvaluation on {name} ({len(y)} labelled accounts) — reproduced by this implementation:")
    for k, v in ev["metrics"].items():
        print(f"  {k:<10} {v:.4f}" if v is not None and not np.isnan(v) else f"  {k:<10} n/a")
    cm = ev["confusion_matrix"]
    print(f"  confusion  TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    if is_demo or model.entry.is_demo:
        print("\nDEMO DATA — NOT REAL SOCIAL MEDIA DATA. Not a research result.")
    if args.out:
        Path(args.out).write_text(json.dumps(json_safe({"model": model.entry.public(), "dataset": name, "evaluation": ev}), indent=2), encoding="utf-8")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
