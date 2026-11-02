"""The humanmock command line: fetch data, train, evaluate, and serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_scores(scores: dict) -> None:
    for part, pair in scores.items():
        print(f"\nTypical error on {part} data, in each measurement's own units:")
        print(f"  {'measurement':34s}{'untrained':>12s}{'trained':>12s}{'count':>8s}")
        for name, row in pair["trained"].items():
            before = pair["untrained"].get(name, {}).get("rmse", float("nan"))
            print(f"  {name:34s}{before:12.3f}{row['rmse']:12.3f}{row['n']:8d}")


def cmd_data(args) -> None:
    root = Path(args.root)
    if args.source in ("nhanes", "all"):
        from humanmock.data.nhanes import run

        print("NHANES: population model for missing parameters")
        result = run(root / "raw" / "nhanes", Path(__file__).parent / "knowledge_data" / "population_nhanes.json")
        print(f"  fitted on {result['n_people']} people")
    if args.source in ("pkdb", "all"):
        from humanmock.data.pkdb import convert, download

        print("PK-DB: drug concentration curves")
        if args.include_closed:
            print("  Including closed-licence studies. Read data/raw/pkdb/TERMS_OF_USE.md and do not redistribute them.")
        download(root / "raw" / "pkdb", args.drugs, args.include_closed)
        convert(root / "raw" / "pkdb", root / "observations" / "pkdb", args.drugs, args.keep_comedication)
    if args.source in ("clinicaltrials", "all"):
        from humanmock.data.clinicaltrials import download as trials

        print("ClinicalTrials.gov: outcomes to review by hand")
        result = trials(root / "raw" / "clinicaltrials", args.condition)
        print(f"  {result['studies']} studies, {result['outcomes']} outcomes, sheet at {result['review_sheet']}")
    if args.source == "synthetic":
        from humanmock.data.synthetic import generate

        print("Synthetic practice data. This is not evidence about real humans.")
        print(" ", generate(root / "observations" / "synthetic", args.subjects, args.seed))
    if args.source == "status":
        from humanmock.data.formats import load_records

        observations = root / "observations"
        records = load_records(observations, include_synthetic=True)
        by_source: dict[str, dict] = {}
        for record in records:
            row = by_source.setdefault(record.source, {"subjects": 0, "measurements": 0, "studies": set()})
            row["subjects"] += 1
            row["measurements"] += len(record.obs_value)
            row["studies"].add(record.study_id)
        for source, row in sorted(by_source.items()):
            print(f"  {source}: {row['subjects']} subjects, {row['measurements']} measurements, {len(row['studies'])} studies")
        if not records:
            print("  No data yet.")


def cmd_train(args) -> None:
    from humanmock.training.train import TrainConfig, run_training

    config = TrainConfig(
        calibration_steps=args.calibration_steps,
        neural_steps=args.neural_steps,
        batch_size=args.batch_size,
        people=args.people,
        learning_rate=args.learning_rate,
        neural_learning_rate=args.neural_learning_rate,
        seed=args.seed,
        eval_every=args.eval_every,
        patience=args.patience,
    )
    meta = run_training(
        Path(args.root) / "observations",
        Path(args.out),
        config,
        members=args.members,
        include_synthetic=args.include_synthetic,
        only_sources=args.sources,
        holdout_studies=args.holdout_study,
        holdout_drugs=args.holdout_drug,
    )
    _print_scores(meta["scores"])


def cmd_evaluate(args) -> None:
    meta = json.loads((Path(args.checkpoint) / "meta.json").read_text(encoding="utf8"))
    print(f"Model {meta['model_id']}, trained {meta['created_at']}")
    print(json.dumps(meta["data"], indent=1))
    _print_scores(meta["scores"])


def cmd_serve(args) -> None:
    import os

    import uvicorn

    if args.checkpoint:
        os.environ["HUMANMOCK_CHECKPOINT"] = args.checkpoint
    os.environ["HUMANMOCK_DB"] = args.database
    uvicorn.run("humanmock.api:create_app", factory=True, host=args.host, port=args.port)


def main() -> None:
    # Show training progress straight away, even when output goes to a log file.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="humanmock", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Download and prepare training data.")
    data.add_argument("source", choices=["all", "nhanes", "pkdb", "clinicaltrials", "synthetic", "status"])
    data.add_argument("--root", default="data")
    data.add_argument("--drugs", nargs="+", default=["paracetamol"])
    data.add_argument("--include-closed", action="store_true", help="Also fetch studies with closed licences.")
    data.add_argument("--keep-comedication", action="store_true", help="Keep curves where other drugs were given too.")
    data.add_argument("--condition", default="streptococcal pharyngitis")
    data.add_argument("--subjects", type=int, default=300)
    data.add_argument("--seed", type=int, default=0)
    data.set_defaults(handler=cmd_data)

    train = sub.add_parser("train", help="Train the body against the data.")
    train.add_argument("--root", default="data")
    train.add_argument("--out", default="checkpoints/run1")
    train.add_argument("--calibration-steps", type=int, default=400)
    train.add_argument("--neural-steps", type=int, default=800)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--people", type=int, default=8, help="Virtual people per subject.")
    train.add_argument("--learning-rate", type=float, default=0.02)
    train.add_argument("--neural-learning-rate", type=float, default=0.002)
    train.add_argument("--members", type=int, default=1, help="Ensemble size. Use 3 or more for real uncertainty.")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--eval-every", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--sources", nargs="+", default=None, help="Only use these folders under data/observations.")
    train.add_argument("--include-synthetic", action="store_true", help="Mix in synthetic practice data.")
    train.add_argument("--holdout-study", nargs="+", default=None, help="Keep these studies out of training.")
    train.add_argument("--holdout-drug", nargs="+", default=None, help="Keep every subject given these drugs out of training.")
    train.set_defaults(handler=cmd_train)

    evaluate = sub.add_parser("evaluate", help="Show how a trained model scored.")
    evaluate.add_argument("checkpoint")
    evaluate.set_defaults(handler=cmd_evaluate)

    serve = sub.add_parser("serve", help="Start the API.")
    serve.add_argument("--checkpoint", default=None, help="Trained model folder. Without it, the untrained body is used.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--database", default="var/sessions.db")
    serve.set_defaults(handler=cmd_serve)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
