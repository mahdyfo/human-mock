"""Fills in parameters the caller did not provide.

A missing value is sampled from people similar to the person described, using a
conditional Gaussian fitted to NHANES. Every virtual person in a session gets
their own sample, so what is unknown shows up as spread in the results rather
than as one misleading average human.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from humanmock.knowledge import KNOWLEDGE, load_json_resource
from humanmock.model import COV_INDEX, COVARIATES

MIN_AGE = float(KNOWLEDGE.patient["age_years"]["min"])
MAX_AGE = float(KNOWLEDGE.patient["age_years"]["max"])
# Hard limits only reject impossible samples and do not describe a healthy range.
LIMITS = {
    name: (float(entry["min"]), float(entry["max"]))
    for name, entry in KNOWLEDGE.patient.items()
    if "min" in entry and name != "age_years"
}
# Parameters NHANES does not cover are sampled independently from a declared prior.
INDEPENDENT = {name: tuple(entry["prior"]) for name, entry in KNOWLEDGE.patient.items() if "prior" in entry}


class PatientError(ValueError):
    """Raised when a described person is outside what this version can simulate."""


class Population:
    def __init__(self, table: dict[str, Any] | None = None):
        table = table or load_json_resource("population_nhanes.json")
        self.source = table["source"]
        self.features = table["features"]
        self.log_features = set(table["log_features"])
        self.age_bands = [tuple(b) for b in table["age_bands"]]
        self.bands = {
            key: (np.array(v["mean"]), np.array(v["cov"])) for key, v in table["bands"].items()
        }
        hist = table["adult_age_histogram"]
        self.age_edges = np.array(hist["edges"], dtype=float)
        self.age_weights = np.array(hist["weights"], dtype=float)
        self.fraction_female = float(table["adult_fraction_female"])
        for name in COVARIATES:
            if name not in ("age_years", "sex_female") and name not in self.features and name not in INDEPENDENT:
                raise ValueError(
                    f"Patient parameter {name} has no way to be sampled. Give it a prior in physiology.yaml, "
                    "or add it to the NHANES fit in humanmock/data/nhanes.py."
                )

    def _band(self, age: float) -> str:
        for low, high in self.age_bands:
            if low <= age < high:
                return f"{low}-{high}"
        return f"{self.age_bands[-1][0]}-{self.age_bands[-1][1]}"

    def _conditional(self, band_key: str, known: dict[str, float], rng, n: int) -> np.ndarray:
        mean, cov = self.bands[band_key]
        known_idx, known_val = [], []
        for j, name in enumerate(self.features):
            if name in known:
                known_idx.append(j)
                value = known[name]
                known_val.append(np.log(value) if name in self.log_features else value)
        missing_idx = [j for j in range(len(self.features)) if j not in known_idx]
        out = np.zeros((n, len(self.features)))
        for j, value in zip(known_idx, known_val):
            out[:, j] = value
        if not missing_idx:
            return out
        if known_idx:
            k = np.array(known_idx)
            m = np.array(missing_idx)
            gain = cov[np.ix_(m, k)] @ np.linalg.pinv(cov[np.ix_(k, k)])
            cond_mean = mean[m] + gain @ (np.array(known_val) - mean[k])
            cond_cov = cov[np.ix_(m, m)] - gain @ cov[np.ix_(k, m)]
        else:
            m = np.array(missing_idx)
            cond_mean, cond_cov = mean[m], cov[np.ix_(m, m)]
        eigval, eigvec = np.linalg.eigh((cond_cov + cond_cov.T) / 2)
        root = eigvec * np.sqrt(np.clip(eigval, 1e-8, None))
        out[:, m] = cond_mean + rng.standard_normal((n, len(m))) @ root.T
        return out

    def sample(self, known: dict[str, Any], n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        """Returns one covariate row per virtual person and a report of what was imputed."""
        known = {k: v for k, v in known.items() if v is not None}
        age = known.get("age_years")
        if age is not None and not (MIN_AGE <= float(age) <= MAX_AGE):
            raise PatientError(f"age_years must be between {MIN_AGE:.0f} and {MAX_AGE:.0f} in this version.")
        for name, value in known.items():
            if name in LIMITS and not (LIMITS[name][0] <= float(value) <= LIMITS[name][1]):
                low, high = LIMITS[name]
                raise PatientError(f"{name}={value} is outside the supported range {low} to {high}.")

        if age is not None:
            ages = np.full(n, float(age))
        else:
            ages = rng.choice(self.age_edges[:-1], size=n, p=self.age_weights) + rng.random(n)
        sex = known.get("sex")
        if sex is not None:
            female = np.full(n, 1.0 if sex == "female" else 0.0)
        else:
            female = (rng.random(n) < self.fraction_female).astype(float)

        cov = np.zeros((n, len(COVARIATES)), dtype=np.float32)
        cov[:, COV_INDEX["age_years"]] = ages
        cov[:, COV_INDEX["sex_female"]] = female

        groups: dict[str, list[int]] = {}
        for i in range(n):
            key = f"{self._band(ages[i])}|{'female' if female[i] else 'male'}"
            groups.setdefault(key, []).append(i)
        known_features = {k: float(known[k]) for k in self.features if k in known}
        for key, rows in groups.items():
            sampled = self._conditional(key, known_features, rng, len(rows))
            for j, name in enumerate(self.features):
                column = sampled[:, j]
                if name in self.log_features:
                    column = np.exp(column)
                cov[rows, COV_INDEX[name]] = column

        for name, (kind, a, b) in INDEPENDENT.items():
            if name in known:
                cov[:, COV_INDEX[name]] = float(known[name])
            elif kind == "normal":
                cov[:, COV_INDEX[name]] = rng.normal(a, b, n)
            else:
                cov[:, COV_INDEX[name]] = np.exp(rng.normal(a, b, n))

        for name, (low, high) in LIMITS.items():
            cov[:, COV_INDEX[name]] = np.clip(cov[:, COV_INDEX[name]], low, high)

        report = {}
        for name in COVARIATES:
            public = "sex" if name == "sex_female" else name
            if public not in known:
                column = cov[:, COV_INDEX[name]]
                report[public] = {"mean": round(float(column.mean()), 3), "sd": round(float(column.std()), 3)}
        return cov, report
