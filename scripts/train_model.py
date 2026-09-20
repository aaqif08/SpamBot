#!/usr/bin/env python
"""Train classifiers for an organisation from the command line (same pipeline as the UI).

    python scripts/train_model.py --email admin@example.com --benchmark cresci-17 --all
    python scripts/train_model.py --email admin@example.com --benchmark combined --algorithm lightgbm --activate
    python scripts/train_model.py --email analyst@example.com --csv path/to/labelled.csv --algorithm xgboost

The user identified by --email must exist (created with `python -m app.cli create-admin`
or through the Users page); everything is attributed to that user's organisation and
recorded in the audit log exactly as if done through the API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.init()

from sqlalchemy import func  # noqa: E402

from app.db.database import SessionLocal  # noqa: E402
from app.db.models import JobType, User  # noqa: E402
from app.services import handlers  # noqa: E402,F401  (registers job handlers)
from app.services.dataset_service import DatasetService  # noqa: E402
from app.services.jobs import create_job, execute_job  # noqa: E402
from app.services.model_service import ModelService, model_public  # noqa: E402
from ml.train import MODEL_ZOO  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", required=True, help="Existing user; training is attributed to their organisation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Labelled CSV to register as a new dataset")
    src.add_argument("--benchmark", choices=["cresci-15", "cresci-17", "combined"], help="Locally installed Cresci benchmark (scripts/fetch_datasets.py)")
    src.add_argument("--dataset-id", help="Existing dataset id")
    p.add_argument("--algorithm", default="lightgbm", choices=sorted(MODEL_ZOO))
    p.add_argument("--all", action="store_true", help="Train every classifier")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--search-iterations", type=int, default=6)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--feature-selection", action="store_true")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activate", action="store_true", help="Promote the (last) trained model to PRODUCTION")
    args = p.parse_args()

    db = SessionLocal()
    try:
        user = db.query(User).filter(func.lower(User.email) == args.email.lower()).first()
        if user is None:
            print(f"No user with email {args.email}. Create one with: python -m app.cli create-admin", file=sys.stderr)
            return 2
        ds_service = DatasetService(db)
        if args.csv:
            path = Path(args.csv)
            if not path.exists():
                print(f"CSV not found: {path}", file=sys.stderr)
                return 2
            ds = ds_service.upload(user, path.read_bytes(), path.name, name=path.stem)
        elif args.benchmark == "combined":
            ds = ds_service.combined_benchmark(user)
        elif args.benchmark:
            ds = ds_service.import_benchmark(user.id, user.organization_id, args.benchmark, progress=lambda m: print("  ", m))
        else:
            ds = ds_service.get(user.organization_id, args.dataset_id)
        version = ds_service.current_version(ds)
        if not version.has_label:
            print("Dataset has no label column; training requires labels.", file=sys.stderr)
            return 2
        print(f"Dataset: {ds.name} v{version.version} ({version.n_rows:,} rows)")

        models = ModelService(db)
        algos = list(MODEL_ZOO) if args.all else [args.algorithm]
        results = []
        for algo in algos:
            row = models.new_training_row(user.organization_id, user.id, algo, None)
            params = {
                "model_row_id": row.id, "dataset_version_id": version.id, "algorithm": algo, "test_size": args.test_size, "cv_folds": args.cv_folds,
                "hyperparameter_search": not args.no_search, "search_iterations": args.search_iterations, "feature_selection": args.feature_selection,
                "feature_selection_top_k": args.top_k, "seed": args.seed, "activate": args.activate and algo == algos[-1], "notes": "trained via scripts/train_model.py",
            }
            job = create_job(db, organization_id=user.organization_id, created_by=user.id, job_type=JobType.TRAINING, params=params, target_type="model", target_id=row.id)
            row.job_id = job.id
            db.commit()
            print(f"Training {MODEL_ZOO[algo].display_name} (job {job.id[:8]})")
            execute_job(job.id)
            db.expire_all()
            row = models.get(user.organization_id, row.id)
            results.append(row)
        print("\nExperiment reproduced by this implementation (hold-out split):")
        print(f"{'model':<22}{'status':<12}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}")
        for r in results:
            m = model_public(r)["test_metrics"]
            if m:
                print(f"{r.name:<22}{r.status.value:<12}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{(m.get('roc_auc') or 0):>8.3f}")
            else:
                print(f"{r.name:<22}{r.status.value:<12}  {r.notes[:60]}")
        prod = models.production(user.organization_id)
        print(f"\nProduction model: {prod.name} v{prod.version} ({prod.id})" if prod else "\nNo production model configured (use --activate or the Models page).")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
