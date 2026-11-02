"""Loads the physiology knowledge base into plain tables the model can use."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import yaml

MAX_DRUGS = 8
"""Fixed number of drug slots. Registry drugs fill the first slots, and sessions
may define hypothetical drugs in the remaining ones."""

DRUG_FIELDS: tuple[str, ...] = (
    "ka",
    "bioavailability",
    "v_l_per_kg",
    "cl_ref_l_h",
    "frac_renal",
    "frac_unbound",
    "kill_emax",
    "kill_ec50",
    "kill_hill",
    "reservoir_penetration",
    "antipyretic_emax",
    "antipyretic_ec50",
    "analgesic_emax",
    "analgesic_ec50",
    "gut_rate",
    "gut_ec50",
    "tox_rate",
    "tox_ec50",
)

# Neutral values keep divisions safe for unused slots and inactive effects.
DRUG_DEFAULTS: dict[str, float] = {
    "ka": 1.0,
    "bioavailability": 1.0,
    "v_l_per_kg": 1.0,
    "cl_ref_l_h": 1.0,
    "frac_renal": 0.0,
    "frac_unbound": 1.0,
    "kill_emax": 0.0,
    "kill_ec50": 1.0,
    "kill_hill": 1.0,
    "reservoir_penetration": 0.0,
    "antipyretic_emax": 0.0,
    "antipyretic_ec50": 1.0,
    "analgesic_emax": 0.0,
    "analgesic_ec50": 1.0,
    "gut_rate": 0.0,
    "gut_ec50": 1.0,
    "tox_rate": 0.0,
    "tox_ec50": 1.0,
}

# Between-person spread applies only to these drug fields.
DRUG_IIV: dict[str, float] = {"ka": 0.3, "v_l_per_kg": 0.2, "cl_ref_l_h": 0.25}


def _read_resource(name: str) -> str:
    return resources.files("humanmock.knowledge_data").joinpath(name).read_text(encoding="utf8")


@dataclass(frozen=True)
class Knowledge:
    raw: dict[str, Any]
    sys_names: tuple[str, ...]
    sys_values: np.ndarray
    sys_iiv: np.ndarray
    path_names: tuple[str, ...]
    path_fields: tuple[str, ...]
    path_values: np.ndarray
    path_iiv: np.ndarray
    drug_names: tuple[str, ...]
    drug_table: np.ndarray
    drug_iiv: np.ndarray

    @property
    def n_pathogens(self) -> int:
        return len(self.path_names)

    def drug_slot(self, name: str) -> int:
        return self.drug_names.index(name)

    def drug_meta(self, name: str) -> dict[str, Any]:
        return self.raw["drugs"][name]


def build_drug_row(spec: dict[str, Any]) -> np.ndarray:
    """Turns a mechanistic drug description into one row of the drug table."""
    row = dict(DRUG_DEFAULTS)
    for key in DRUG_FIELDS:
        if key in spec and spec[key] is not None:
            row[key] = float(spec[key])
    return np.array([row[k] for k in DRUG_FIELDS], dtype=np.float32)


def load_knowledge() -> Knowledge:
    raw = yaml.safe_load(_read_resource("physiology.yaml"))
    sys_names = tuple(raw["system"].keys())
    sys_values = np.array([raw["system"][k]["value"] for k in sys_names], dtype=np.float32)
    sys_iiv = np.array([raw["system"][k]["iiv"] for k in sys_names], dtype=np.float32)

    path_names = tuple(raw["pathogens"].keys())
    first = raw["pathogens"][path_names[0]]
    path_fields = tuple(k for k, v in first.items() if isinstance(v, dict))
    path_values = np.array(
        [[raw["pathogens"][p][f]["value"] for f in path_fields] for p in path_names], dtype=np.float32
    )
    path_iiv = np.array(
        [[raw["pathogens"][p][f]["iiv"] for f in path_fields] for p in path_names], dtype=np.float32
    )

    drug_names = tuple(raw["drugs"].keys())
    if len(drug_names) > MAX_DRUGS:
        raise ValueError(f"Registry has {len(drug_names)} drugs but MAX_DRUGS is {MAX_DRUGS}.")
    table = np.tile(build_drug_row({}), (MAX_DRUGS, 1))
    for i, name in enumerate(drug_names):
        table[i] = build_drug_row(raw["drugs"][name])
    iiv = np.array([DRUG_IIV.get(k, 0.0) for k in DRUG_FIELDS], dtype=np.float32)

    return Knowledge(
        raw=raw,
        sys_names=sys_names,
        sys_values=sys_values,
        sys_iiv=sys_iiv,
        path_names=path_names,
        path_fields=path_fields,
        path_values=path_values,
        path_iiv=path_iiv,
        drug_names=drug_names,
        drug_table=table,
        drug_iiv=iiv,
    )


KNOWLEDGE = load_knowledge()


def load_json_resource(name: str) -> dict[str, Any]:
    return json.loads(_read_resource(name))
