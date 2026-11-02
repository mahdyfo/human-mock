"""The humanmock command line: start the app, fetch data, train, evaluate, and serve."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
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
    if args.source == "check":
        from humanmock.data.formats import check_source

        observations = root / "observations"
        folders = [Path(args.target)] if args.target else sorted(p for p in observations.iterdir() if p.is_dir())
        failed = False
        for folder in folders:
            report = check_source(folder)
            print(f"{report.name}: {report.subjects} subjects, {report.measurements} measurements")
            for error in report.errors:
                failed = True
                print(f"  ERROR: {error}")
            for note in report.notes:
                print(f"  note: {note}")
        if failed:
            raise SystemExit(1)
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


def cmd_evidence(args) -> None:
    from humanmock.knowledge import RAW, evidence_gaps

    gaps, total = evidence_gaps(RAW)
    print(f"{len(gaps)} of {total} constants in physiology.yaml have no cited source yet.")
    print("Replacing one with a cited value is a useful first contribution. See CONTRIBUTING.md.\n")
    for path, note in gaps:
        print(f"  {path}: {note}")


def _run_server(checkpoint: str | None, host: str, port: int, database: str) -> None:
    import os

    import uvicorn

    if checkpoint:
        os.environ["HUMANMOCK_CHECKPOINT"] = checkpoint
    os.environ["HUMANMOCK_DB"] = database
    uvicorn.run("humanmock.api:create_app", factory=True, host=host, port=port)


def cmd_serve(args) -> None:
    _run_server(args.checkpoint, args.host, args.port, args.database)


def newest_checkpoint(root: Path) -> Path | None:
    """The most recent trained model that fits the body as it is now."""
    from humanmock.checkpoint import structure_fingerprint

    fingerprint = structure_fingerprint()
    found = sorted(root.glob("*/meta.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for meta in found:
        try:
            if json.loads(meta.read_text(encoding="utf8")).get("structure") == fingerprint:
                return meta.parent
        except (OSError, ValueError):
            continue
    return None


def free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """The preferred port if it is free, and otherwise any free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise SystemExit("No network port is free on this computer.")


def open_when_ready(url: str, attempts: int = 240) -> None:
    """Opens the page once the server answers, because a browser opened sooner shows an error."""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=2):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)


def cmd_start(args) -> None:
    host = "127.0.0.1"
    port = free_port(args.port, host)
    url = f"http://{host}:{port}"
    checkpoint = newest_checkpoint(Path(args.checkpoints))
    if checkpoint:
        print(f"Using the trained model in {checkpoint}.")
    else:
        print("Using the untrained body, because no trained model was found. docs/TRAINING.md explains how to train one.")
    print(f"humanmock is starting at {url}. Press Ctrl+C here to stop it.")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    _run_server(str(checkpoint) if checkpoint else None, host, port, args.database)


def main() -> None:
    # Show training progress straight away, even when output goes to a log file.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="humanmock", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the app and open it in your browser.")
    start.add_argument("--port", type=int, default=8000, help="Used if free. Otherwise any free port is chosen.")
    start.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    start.add_argument("--checkpoints", default="checkpoints", help="Folder to look in for the newest trained model.")
    start.add_argument("--database", default="var/sessions.db")
    start.set_defaults(handler=cmd_start)

    data = sub.add_parser("data", help="Download, prepare, and check training data.")
    data.add_argument("source", choices=["all", "nhanes", "pkdb", "clinicaltrials", "synthetic", "status", "check"])
    data.add_argument("target", nargs="?", default=None, help="With check, the one folder to check.")
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

    evidence = sub.add_parser("evidence", help="List constants that still need a cited source.")
    evidence.set_defaults(handler=cmd_evidence)

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
