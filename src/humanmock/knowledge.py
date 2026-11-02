"""Loads the physiology knowledge base into plain tables the model can use.

The file is checked as it loads, so a misspelled field stops with a clear message
rather than silently becoming a default.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import yaml

SPARE_DRUG_SLOTS = 4
"""Slots kept free so a session can define hypothetical drugs."""

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
DRUG_DETAILS = ("label", "sources", "max_single_dose_mg", "max_daily_dose_mg")
PRIOR_KINDS = ("normal", "lognormal")
UNCITED = ("assumption", "not re-verified")


class KnowledgeError(ValueError):
    """Raised when physiology.yaml is incomplete or misspelled."""


def _read_resource(name: str) -> str:
    return resources.files("humanmock.knowledge_data").joinpath(name).read_text(encoding="utf8")


def _check_constant(path: str, entry: Any, problems: list[str]) -> None:
    for key in ("value", "iiv", "source", "note"):
        if not isinstance(entry, dict) or key not in entry:
            problems.append(f"{path} needs a {key}.")


def _did_you_mean(name: str, options: tuple[str, ...]) -> str:
    close = difflib.get_close_matches(name, options, n=1)
    return f" Did you mean {close[0]}?" if close else ""


def validate_knowledge(raw: dict[str, Any]) -> list[str]:
    """Returns every problem in the knowledge base, in plain language."""
    problems: list[str] = []
    for name, entry in raw["patient"].items():
        for key in ("center", "scale", "note"):
            if key not in entry:
                problems.append(f"patient.{name} needs a {key}.")
        if "prior" in entry:
            prior = entry["prior"]
            if len(prior) != 3 or prior[0] not in PRIOR_KINDS:
                problems.append(f"patient.{name} prior must be [normal, mean, sd] or [lognormal, log mean, log sd].")
            if "source" not in entry:
                problems.append(f"patient.{name} has a prior, so it needs a source.")
    for name, entry in raw["system"].items():
        _check_constant(f"system.{name}", entry, problems)

    fields: set[str] | None = None
    for name, pathogen in raw["pathogens"].items():
        if "label" not in pathogen:
            problems.append(f"pathogens.{name} needs a label.")
        own = {k for k, v in pathogen.items() if k != "label"}
        for field in own:
            _check_constant(f"pathogens.{name}.{field}", pathogen[field], problems)
        if fields is None:
            fields = own
        for field in sorted(fields - own):
            problems.append(f"pathogens.{name} is missing {field}, which every pathogen needs.")
        for field in sorted(own - fields):
            problems.append(f"pathogens.{name}.{field} is not a pathogen field.{_did_you_mean(field, tuple(fields))}")

    known = DRUG_FIELDS + DRUG_DETAILS
    for name, drug in raw["drugs"].items():
        for key in ("label", "sources"):
            if key not in drug:
                problems.append(f"drugs.{name} needs {key}.")
        for key in drug:
            if key not in known:
                problems.append(f"drugs.{name}.{key} is not a drug field.{_did_you_mean(key, known)}")
    return problems


def evidence_gaps(raw: dict[str, Any]) -> tuple[list[tuple[str, str]], int]:
    """Lists constants with no cited source, and how many constants there are in total."""
    gaps: list[tuple[str, str]] = []
    total = 0

    def visit(path: str, source: Any, note: str) -> None:
        nonlocal total
        total += 1
        if any(word in str(source).lower() for word in UNCITED):
            gaps.append((path, note))

    for name, entry in raw["patient"].items():
        if "prior" in entry:
            visit(f"patient.{name}", entry["source"], f"{entry['note']}: how missing values are sampled")
    for name, entry in raw["system"].items():
        visit(f"system.{name}", entry["source"], entry["note"])
    for name, pathogen in raw["pathogens"].items():
        for field, entry in pathogen.items():
            if field != "label":
                visit(f"pathogens.{name}.{field}", entry["source"], entry["note"])
    for name, drug in raw["drugs"].items():
        for topic, source in drug["sources"].items():
            visit(f"drugs.{name}.{topic}", source, str(source) if str(source) != "assumption" else f"{topic} of {name}")
    return gaps, total


@dataclass(frozen=True)
class Knowledge:
    raw: dict[str, Any]
    patient: dict[str, dict[str, Any]]
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


RAW: dict[str, Any] = yaml.safe_load(_read_resource("physiology.yaml"))

MAX_DRUGS = len(RAW["drugs"]) + SPARE_DRUG_SLOTS
"""Number of drug slots. Named drugs fill the first slots, and sessions may define
hypothetical drugs in the rest. It grows by itself as drugs are added."""


def load_knowledge(raw: dict[str, Any] = RAW) -> Knowledge:
    problems = validate_knowledge(raw)
    if problems:
        raise KnowledgeError("physiology.yaml has problems:\n- " + "\n- ".join(problems))
    sys_names = tuple(raw["system"].keys())
    sys_values = np.array([raw["system"][k]["value"] for k in sys_names], dtype=np.float32)
    sys_iiv = np.array([raw["system"][k]["iiv"] for k in sys_names], dtype=np.float32)

    path_names = tuple(raw["pathogens"].keys())
    first = raw["pathogens"][path_names[0]]
    path_fields = tuple(k for k in first if k != "label")
    path_values = np.array(
        [[raw["pathogens"][p][f]["value"] for f in path_fields] for p in path_names], dtype=np.float32
    )
    path_iiv = np.array(
        [[raw["pathogens"][p][f]["iiv"] for f in path_fields] for p in path_names], dtype=np.float32
    )

    drug_names = tuple(raw["drugs"].keys())
    table = np.tile(build_drug_row({}), (MAX_DRUGS, 1))
    for i, name in enumerate(drug_names):
        table[i] = build_drug_row(raw["drugs"][name])
    iiv = np.array([DRUG_IIV.get(k, 0.0) for k in DRUG_FIELDS], dtype=np.float32)

    return Knowledge(
        raw=raw,
        patient=raw["patient"],
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
