"""Saving and loading trained models."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from humanmock.engine import Simulator
from humanmock.knowledge import KNOWLEDGE
from humanmock.model import LAYOUT, OBS_NAMES, BodyModel


def structure_fingerprint() -> str:
    """Changes whenever organs or the knowledge tables change shape."""
    text = json.dumps(
        {
            "states": [(e[0], e[2]) for e in LAYOUT.entries],
            "size": LAYOUT.size,
            "observables": list(OBS_NAMES),
            "system": list(KNOWLEDGE.sys_names),
            "pathogens": list(KNOWLEDGE.path_names),
            "pathogen_fields": list(KNOWLEDGE.path_fields),
            "drugs": list(KNOWLEDGE.drug_names),
        }
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save_checkpoint(directory: str | Path, members: list[BodyModel], meta: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(members):
        eqx.tree_serialise_leaves(directory / f"member_{i}.eqx", model)
    meta = {
        **meta,
        "status": "trained",
        "model_id": meta.get("model_id") or uuid.uuid4().hex[:12],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ensemble_size": len(members),
        "structure": structure_fingerprint(),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    return directory


def load_members(directory: str | Path) -> tuple[list[BodyModel], dict[str, Any]]:
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No trained model at {directory}. Expected a meta.json file there.")
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    if meta.get("structure") != structure_fingerprint():
        raise ValueError(
            "This checkpoint was trained on a different body structure, because organs or knowledge tables "
            "have changed since. Train a new model, or use the code version that produced it."
        )
    members = []
    for i in range(int(meta["ensemble_size"])):
        skeleton = BodyModel.create(jax.random.PRNGKey(0))
        members.append(eqx.tree_deserialise_leaves(directory / f"member_{i}.eqx", skeleton))
    return members, meta


def load_simulator(directory: str | Path) -> Simulator:
    members, meta = load_members(directory)
    return Simulator(members=members, meta=meta)
