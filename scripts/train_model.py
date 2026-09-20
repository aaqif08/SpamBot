#!/usr/bin/env python
"""Train one (or all) classifiers on a registered dataset, a Cresci dataset or a CSV.

Examples
--------
    python scripts/train_model.py --dataset demo --algorithm lightgbm
    python scripts/train_model.py --dataset cresci-15 --all
    python scripts/train_model.py --csv path/to/labelled.csv --algorithm xgboost --feature-selection --top-k 20

Results are written to backend/models/<model_id>/ and registered in
backend/models/registry.json and the SQLite ``models`` table. Metrics printed
here are *measured by this implementation* — never the paper's numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path setup)

_bootstrap.init()

from app.db.database import SessionLocal  # noqa: E402
from app.services.dataset_service import DatasetService  # noqa: E402
from app.services.registry import get_registry  # noqa: E402
from app.services.training_service import sync_model_records  # noqa: E402
from ml.datasets import labelled_frame, read_csv_safely  # noqa: E402
from ml.train import MODEL_ZOO, TrainConfig, train_all_models, train_model  # noqa: E402


def resolve_dataset(args: argparse.Namespace):
    db = SessionLocal()
    service = DatasetService(db)
    try:
        if args.csv:
            path = Path(args.csv)
            if not path.exists():
                sys.exit(f"CSV not found: {path}")
            info = service.save_upload(path.read_bytes(), path.name, name=path.stem)
            row, df = service.load_frame(info["id"])
        elif args.dataset == "demo":
            info = service.create_demo(n=args.demo_size, seed=args.seed)
            row, df = service.load_frame(info["id"])
        elif args.dataset in ("cresci-15", "cresci-17"):
            status = service.cresci_status(args.dataset)
            if not status["imported"] or args.reimport:
                print(f"Importing {args.dataset} (this can take a while)…")
                info = service.import_cresci(args.dataset, progress=lambda m: print("  ", m), use_cache=not args.reimport)
            else:
                info = {"id": status["dataset_id"]}
            row, df = service.load_frame(info["id"])
        elif args.dataset == "cresci-combined":
            info = service.create_cresci_combined(progress=lambda m: print("  ", m))
            row, df = service.load_frame(info["id"])
        else:
            row, df = service.load_frame(args.dataset)
        return row, df
    finally:
        db.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cresci-17", help="cresci-15 | cresci-17 | cresci-combined | demo | <dataset id>")
    p.add_argument("--csv", help="Path to a labelled CSV (registered as an upload)")
    p.add_argument("--algorithm", default="lightgbm", choices=sorted(MODEL_ZOO))
    p.add_argument("--all", action="store_true", help="Train every classifier in the zoo")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-search", action="store_true", help="Disable hyperparameter search")
    p.add_argument("--search-iterations", type=int, default=6)
    p.add_argument("--feature-selection", action="store_true", help="SHAP-based feature selection (paper §III-C)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--demo-size", type=int, default=600)
    p.add_argument("--reimport", action="store_true", help="Rebuild Cresci feature cache")
    p.add_argument("--no-activate", action="store_true")
    args = p.parse_args()

    row, df = resolve_dataset(args)
    X, y = labelled_frame(df)
    print(f"Dataset: {row.name} ({len(X)} labelled accounts, demo={row.is_demo})")

    base = TrainConfig(
        algorithm=args.algorithm,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        seed=args.seed,
        hyperparameter_search=not args.no_search,
        search_iterations=args.search_iterations,
        feature_selection=args.feature_selection,
        feature_selection_top_k=args.top_k,
        dataset_id=row.id,
        dataset_name=row.name,
        is_demo=bool(row.is_demo),
        activate=not args.no_activate,
    )
    registry = get_registry()

    def progress(stage: str, pct: float, msg: str) -> None:
        print(f"  [{stage:<19}] {pct:>4.0%}  {msg}")

    if args.all:
        results = train_all_models(X, y, base, registry, progress=progress)
    else:
        results = [train_model(X, y, base, registry, progress=progress)]

    db = SessionLocal()
    try:
        sync_model_records(db)
    finally:
        db.close()

    print("\nExperiment reproduced by this implementation (hold-out split):")
    print(f"{'model':<22}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}{'time':>9}")
    for r in results:
        m = r.entry.metrics["holdout"]
        print(f"{r.entry.name:<22}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{(m['roc_auc'] or float('nan')):>8.3f}{r.entry.training_seconds:>8.1f}s")
    if row.is_demo:
        print("\nDEMO DATA — NOT REAL SOCIAL MEDIA DATA. These numbers are not research results.")
    print(f"\nActive model: {registry.active_id()}")
    print(json.dumps({"models": [r.model_id for r in results]}))


if __name__ == "__main__":
    main()
