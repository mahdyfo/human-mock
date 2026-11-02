"""Downloads NHANES and fits the population model that fills in missing parameters.

NHANES is the US national health examination survey. Its public files are free
to download. This module turns them into means and covariances by age band and
sex, so a missing value is sampled from people similar to the person described.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt"
FILES = {
    "DEMO_J": ["RIAGENDR", "RIDAGEYR", "WTMEC2YR"],
    "BMX_J": ["BMXWT", "BMXHT"],
    "BIOPRO_J": ["LBXSCR", "LBXSATSI"],
    "CBC_J": ["LBXWBCSI"],
    "BPX_J": ["BPXPLS", "BPXSY1", "BPXDI1"],
    "HSCRP_J": ["LBXHSCRP"],
}
RENAME = {
    "RIAGENDR": "sex",
    "RIDAGEYR": "age_years",
    "WTMEC2YR": "survey_weight",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "LBXSCR": "creatinine_mg_dl",
    "LBXSATSI": "alt_u_l",
    "LBXWBCSI": "wbc_k_ul",
    "BPXPLS": "heart_rate_bpm",
    "BPXSY1": "systolic_bp_mmhg",
    "BPXDI1": "diastolic_bp_mmhg",
    "LBXHSCRP": "crp_mg_l",
}
# Skewed measurements are modelled on a log scale.
LOG_FEATURES = ("weight_kg", "height_cm", "creatinine_mg_dl", "alt_u_l", "wbc_k_ul", "crp_mg_l")
LINEAR_FEATURES = ("heart_rate_bpm", "systolic_bp_mmhg", "diastolic_bp_mmhg")
FEATURES = LOG_FEATURES + LINEAR_FEATURES
AGE_BANDS = ((12, 20), (20, 40), (40, 60), (60, 70), (70, 81))


def download(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": "humanmock"}) as client:
        for name in FILES:
            path = raw_dir / f"{name}.xpt"
            if path.exists() and path.stat().st_size > 10_000:
                print(f"  have {path.name}")
            else:
                print(f"  downloading {path.name}")
                response = client.get(BASE_URL.format(name=name))
                response.raise_for_status()
                path.write_bytes(response.content)
            paths.append(path)
    return paths


def load_table(raw_dir: Path) -> pd.DataFrame:
    table = None
    for name, columns in FILES.items():
        frame = pd.read_sas(raw_dir / f"{name}.xpt", format="xport")[["SEQN", *columns]]
        table = frame if table is None else table.merge(frame, on="SEQN", how="left")
    table = table.rename(columns=RENAME)
    return table[table["age_years"] >= 12].copy()


def _weighted_fit(frame: pd.DataFrame) -> dict:
    values = np.full((len(frame), len(FEATURES)), np.nan)
    for j, name in enumerate(FEATURES):
        column = frame[name].to_numpy(dtype=float)
        if name in LOG_FEATURES:
            column = np.log(np.where(column > 0, column, np.nan))
        else:
            column = np.where(column > 0, column, np.nan)
        values[:, j] = column
    weights = frame["survey_weight"].to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    mean = np.zeros(len(FEATURES))
    for j in range(len(FEATURES)):
        ok = np.isfinite(values[:, j]) & (weights > 0)
        mean[j] = np.average(values[ok, j], weights=weights[ok])

    # Pairwise covariance keeps every usable measurement when some are missing.
    cov = np.zeros((len(FEATURES), len(FEATURES)))
    for a in range(len(FEATURES)):
        for b in range(a, len(FEATURES)):
            ok = np.isfinite(values[:, a]) & np.isfinite(values[:, b]) & (weights > 0)
            cov[a, b] = cov[b, a] = np.average(
                (values[ok, a] - mean[a]) * (values[ok, b] - mean[b]), weights=weights[ok]
            )
    # Pairwise estimates can be slightly indefinite, so nudge the matrix to be usable.
    eigval, eigvec = np.linalg.eigh(cov)
    cov = (eigvec * np.clip(eigval, 1e-4, None)) @ eigvec.T
    return {"n": int(len(frame)), "mean": mean.tolist(), "cov": cov.tolist()}


def fit(raw_dir: Path) -> dict:
    table = load_table(raw_dir)
    bands = {}
    for low, high in AGE_BANDS:
        for sex_code, sex in ((1, "male"), (2, "female")):
            subset = table[(table["age_years"] >= low) & (table["age_years"] < high) & (table["sex"] == sex_code)]
            bands[f"{low}-{high}|{sex}"] = _weighted_fit(subset)

    adults = table[table["age_years"] >= 18]
    ages = adults["age_years"].to_numpy(dtype=float)
    weights = adults["survey_weight"].fillna(0).to_numpy(dtype=float)
    histogram, edges = np.histogram(ages, bins=np.arange(18, 82), weights=weights)
    female = float(np.average(adults["sex"].to_numpy() == 2, weights=weights))
    return {
        "source": "NHANES 2017-2018 public files, survey weighted, https://wwwn.cdc.gov/nchs/nhanes/",
        "features": list(FEATURES),
        "log_features": list(LOG_FEATURES),
        "age_bands": [list(b) for b in AGE_BANDS],
        "bands": bands,
        "adult_age_histogram": {"edges": edges.tolist(), "weights": (histogram / histogram.sum()).tolist()},
        "adult_fraction_female": female,
        "n_people": int(len(table)),
    }


def run(raw_dir: Path, output: Path) -> dict:
    download(raw_dir)
    model = fit(raw_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed line endings, so refitting on any system leaves this tracked file unchanged.
    output.write_text(json.dumps(model, indent=1), encoding="utf8", newline="\n")
    return model
