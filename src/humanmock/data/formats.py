"""The one data format every training source is converted into.

A source is a folder under data/observations/ holding three CSV files.

subjects.csv     one row per person or trial arm
events.csv       doses and exposures, with time_h measured from time zero
observations.csv what was measured, and when

Time zero is the moment of the first treatment. A subject's conditions column
says how far into an illness they were at that moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from humanmock.knowledge import KNOWLEDGE, MAX_DRUGS
from humanmock.model import COVARIATES, OBS_INDEX
from humanmock.population import Population

KNOWN_COVARIATES = tuple(c for c in COVARIATES if c != "sex_female") + ("sex",)
NO_EXPOSURE = -10.0


class DataFormatError(ValueError):
    pass


def variable_index(name: str) -> int:
    """Maps a column value such as conc_amoxicillin_mg_l to an observable."""
    if name.startswith("conc_") and name.endswith("_mg_l"):
        drug = name[len("conc_") : -len("_mg_l")]
        if drug not in KNOWLEDGE.drug_names:
            raise DataFormatError(f"Observation names unknown drug '{drug}'. Add it to physiology.yaml first.")
        return OBS_INDEX[f"conc_slot{KNOWLEDGE.drug_slot(drug)}"]
    if name not in OBS_INDEX or name.startswith("conc_slot"):
        known = sorted(n for n in OBS_INDEX if not n.startswith("conc_slot"))
        raise DataFormatError(f"Unknown variable '{name}'. Use conc_<drug>_mg_l or one of {known}.")
    return OBS_INDEX[name]


@dataclass
class Record:
    subject_id: str
    study_id: str
    source: str
    split: str
    is_arm: bool
    known: dict[str, Any]
    path_slot: int
    onset_days: float
    times: np.ndarray
    dt: np.ndarray
    oral: np.ndarray
    iv: np.ndarray
    exposure: np.ndarray
    obs_index: np.ndarray
    obs_var: np.ndarray
    obs_value: np.ndarray
    obs_sd: np.ndarray
    drugs_seen: dict[str, list[float]] = field(default_factory=dict)
    cov: np.ndarray | None = None


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()


def read_source(folder: Path) -> list[Record]:
    for name in ("subjects.csv", "events.csv", "observations.csv"):
        if not (folder / name).exists():
            return []
    subjects = pd.read_csv(folder / "subjects.csv", dtype={"subject_id": str, "study_id": str})
    events = pd.read_csv(folder / "events.csv", dtype={"subject_id": str})
    observations = pd.read_csv(folder / "observations.csv", dtype={"subject_id": str})
    if subjects.empty:
        return []
    for frame, required, label in (
        (subjects, ("subject_id", "study_id"), "subjects.csv"),
        (events, ("subject_id", "time_h", "type"), "events.csv"),
        (observations, ("subject_id", "time_h", "variable", "value"), "observations.csv"),
    ):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise DataFormatError(f"{folder / label} is missing columns {missing}.")
    if subjects["subject_id"].duplicated().any():
        raise DataFormatError(f"{folder / 'subjects.csv'} repeats a subject_id.")

    events_by = dict(tuple(events.groupby("subject_id")))
    obs_by = dict(tuple(observations.groupby("subject_id")))
    records = []
    for _, row in subjects.iterrows():
        sid = row["subject_id"]
        ev = events_by.get(sid, events.iloc[0:0])
        ob = obs_by.get(sid, observations.iloc[0:0])
        if ob.empty:
            continue

        known: dict[str, Any] = {}
        for name in KNOWN_COVARIATES:
            if name in row and pd.notna(row[name]) and _text(row[name]) != "":
                known[name] = row[name] if name == "sex" else float(row[name])

        path_slot, onset_days = -1, 0.0
        conditions = _text(row.get("conditions"))
        if conditions:
            name, _, days = conditions.partition(":")
            if name not in KNOWLEDGE.path_names:
                raise DataFormatError(f"Subject {sid} has unknown condition '{name}'.")
            path_slot = KNOWLEDGE.path_names.index(name)
            onset_days = float(days or 1.0)

        times = sorted({0.0, *ev["time_h"].astype(float), *ob["time_h"].astype(float)})
        index = {t: i for i, t in enumerate(times)}
        T = len(times)
        oral = np.zeros((T, MAX_DRUGS), dtype=np.float32)
        iv = np.zeros((T, MAX_DRUGS), dtype=np.float32)
        exposure = np.full((T, KNOWLEDGE.n_pathogens), NO_EXPOSURE, dtype=np.float32)
        drugs_seen: dict[str, list[float]] = {}
        for _, event in ev.iterrows():
            i = index[float(event["time_h"])]
            if event["type"] == "dose":
                drug = _text(event.get("drug"))
                if drug not in KNOWLEDGE.drug_names:
                    raise DataFormatError(f"Subject {sid} was given unknown drug '{drug}'.")
                slot = KNOWLEDGE.drug_slot(drug)
                mg = float(event["dose_mg"])
                (iv if _text(event.get("route")) == "iv" else oral)[i, slot] += mg
                drugs_seen.setdefault(drug, []).append(mg)
            elif event["type"] == "exposure":
                pathogen = _text(event.get("pathogen"))
                if pathogen not in KNOWLEDGE.path_names:
                    raise DataFormatError(f"Subject {sid} was exposed to unknown pathogen '{pathogen}'.")
                slot = KNOWLEDGE.path_names.index(pathogen)
                field_i = KNOWLEDGE.path_fields.index("log10_inoculum")
                exposure[i, slot] = float(KNOWLEDGE.path_values[slot, field_i])
            else:
                raise DataFormatError(f"Subject {sid} has unknown event type '{event['type']}'.")

        split = _text(row.get("split")) or "auto"
        if split not in ("auto", "train", "val", "holdout"):
            raise DataFormatError(f"Subject {sid} has unknown split '{split}'.")
        sd = ob["sd"].to_numpy(dtype=np.float32) if "sd" in ob.columns else np.full(len(ob), np.nan, dtype=np.float32)
        records.append(
            Record(
                subject_id=sid,
                study_id=_text(row["study_id"]),
                source=folder.name,
                split=split,
                is_arm=_text(row.get("kind")) == "arm",
                known=known,
                path_slot=path_slot,
                onset_days=onset_days,
                times=np.array(times, dtype=np.float32),
                dt=np.diff(np.array([0.0] + times, dtype=np.float32)),
                oral=oral,
                iv=iv,
                exposure=exposure,
                obs_index=np.array([index[float(t)] for t in ob["time_h"]], dtype=np.int32),
                obs_var=np.array([variable_index(v) for v in ob["variable"]], dtype=np.int32),
                obs_value=ob["value"].to_numpy(dtype=np.float32),
                obs_sd=np.nan_to_num(sd, nan=0.0),
                drugs_seen=drugs_seen,
            )
        )
    return records


def load_records(root: Path, include_synthetic: bool = False, only: list[str] | None = None) -> list[Record]:
    records: list[Record] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if only is not None and folder.name not in only:
            continue
        if only is None and folder.name == "synthetic" and not include_synthetic:
            continue
        records.extend(read_source(folder))
    return records


def sample_people(records: list[Record], people: int, seed: int, population: Population | None = None) -> None:
    """Gives each record its own virtual people, filling in whatever the source did not report."""
    population = population or Population()
    rng = np.random.default_rng(seed)
    for record in records:
        known = dict(record.known)
        if "age_years" not in known:
            known["age_years"] = float(rng.uniform(20.0, 45.0))
        record.cov, _ = population.sample(known, people, rng)


def assign_splits(
    records: list[Record],
    seed: int,
    val_fraction: float = 0.2,
    holdout_studies: list[str] | None = None,
    holdout_drugs: list[str] | None = None,
) -> None:
    """Splits by whole study, so a score never comes from people the model has already met."""
    holdout_studies = holdout_studies or []
    holdout_drugs = holdout_drugs or []
    rng = np.random.default_rng(seed)
    by_source: dict[str, list[str]] = {}
    for record in records:
        if record.split == "auto":
            by_source.setdefault(record.source, [])
            if record.study_id not in by_source[record.source]:
                by_source[record.source].append(record.study_id)
    val_studies: set[str] = set()
    for studies in by_source.values():
        order = list(rng.permutation(sorted(studies)))
        count = int(round(len(order) * val_fraction))
        if len(order) >= 2:
            count = max(count, 1)
        val_studies.update(order[:count])
    for record in records:
        if record.study_id in holdout_studies or any(d in record.drugs_seen for d in holdout_drugs):
            record.split = "holdout"
        elif record.split == "auto":
            record.split = "val" if record.study_id in val_studies else "train"


def support_summary(records: list[Record]) -> dict[str, Any]:
    """What the model has actually seen, so the API can warn when asked about anything else."""
    from humanmock.model import COV_INDEX

    train = [r for r in records if r.split == "train"]
    conditions = sorted({KNOWLEDGE.path_names[r.path_slot] for r in train if r.path_slot >= 0})
    doses: dict[str, list[float]] = {}
    for record in train:
        for drug, amounts in record.drugs_seen.items():
            doses.setdefault(drug, []).extend(amounts)
    columns = {}
    for name in ("age_years", "weight_kg", "creatinine_mg_dl"):
        values = np.concatenate([r.cov[:, COV_INDEX[name]] for r in train]) if train else np.array([])
        if len(values):
            columns[name] = (float(np.percentile(values, 1)), float(np.percentile(values, 99)))
    return {
        "conditions": conditions,
        "drugs": {d: (min(v), max(v)) for d, v in doses.items()},
        "covariates": columns,
        "subjects": len(train),
    }
