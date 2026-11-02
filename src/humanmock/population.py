"""Fills in parameters the caller did not provide.

A missing value is sampled from people similar to the person described, using a
conditional Gaussian fitted to NHANES. Every virtual person in a session gets
their own sample, so what is unknown shows up as spread in the results rather
than as one misleading average human.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from humanmock.knowledge import load_json_resource
from humanmock.model import COV_INDEX, COVARIATES

MIN_AGE = 12.0
MAX_AGE = 90.0
# Hard limits only reject impossible samples and do not describe a healthy range.
LIMITS = {
    "weight_kg": (25.0, 250.0),
    "height_cm": (120.0, 220.0),
    "creatinine_mg_dl": (0.2, 12.0),
    "alt_u_l": (3.0, 400.0),
    "wbc_k_ul": (1.5, 25.0),
    "crp_mg_l": (0.05, 60.0),
    "heart_rate_bpm": (40.0, 130.0),
    "systolic_bp_mmhg": (70.0, 220.0),
    "diastolic_bp_mmhg": (30.0, 130.0),
    "respiratory_rate_bpm": (8.0, 30.0),
    "body_temp_c": (35.5, 37.8),
    "immune_function": (0.05, 2.0),
    "hepatic_function": (0.05, 1.5),
}
# Not measured by NHANES, so these spreads are assumptions.
INDEPENDENT = {
    "respiratory_rate_bpm": ("normal", 15.0, 2.0),
    "body_temp_c": ("normal", 36.7, 0.25),
    "immune_function": ("lognormal", 0.0, 0.2),
    "hepatic_function": ("lognormal", 0.0, 0.1),
}


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
