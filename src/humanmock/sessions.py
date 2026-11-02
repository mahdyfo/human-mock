"""Session memory: every virtual body and its history, kept on disk.

A session keeps a snapshot of its virtual people after every step, including
their immune memory and their memory neurons. That makes four things possible.

Continuing: the next call picks up exactly where the last one stopped, even
after the server restarts.
Resetting: go back to the start, or to any earlier step, and forget what came
after it.
Forking: copy a session and try a different treatment on identical bodies.
Reviewing: read back everything that happened.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from humanmock.engine import DrugBook, Particles, Simulator, expand_events, patient_summary
from humanmock.knowledge import KNOWLEDGE
from humanmock.model import COV_INDEX
from humanmock.population import PatientError

MAX_STEP_HOURS = 24.0 * 90
MAX_EVENTS_PER_STEP = 500
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(m|min|h|hr|hours?|d|days?)?\s*$", re.IGNORECASE)


class SessionNotFound(KeyError):
    pass


class StepNotFound(KeyError):
    pass


class ModelMismatch(RuntimeError):
    pass


def parse_duration(value: str | float | int) -> float:
    """Accepts a number of hours, or text such as 30m, 8h, or 3d."""
    if isinstance(value, (int, float)):
        hours = float(value)
    else:
        match = _DURATION.match(str(value))
        if not match:
            raise PatientError(f"Cannot read duration '{value}'. Use a form such as 30m, 8h, or 3d.")
        amount = float(match.group(1))
        unit = (match.group(2) or "h").lower()
        hours = amount / 60.0 if unit.startswith("m") else amount * 24.0 if unit.startswith("d") else amount
    if hours <= 0:
        raise PatientError("Duration must be greater than zero.")
    if hours > MAX_STEP_HOURS:
        raise PatientError(f"One step can cover at most {MAX_STEP_HOURS / 24:.0f} days.")
    return hours


def _encode(p: Particles) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **p.to_arrays())
    return buffer.getvalue()


def _decode(blob: bytes) -> Particles:
    with np.load(io.BytesIO(blob)) as data:
        return Particles.from_arrays({k: data[k] for k in data.files})


class SessionStore:
    """Plain SQLite storage, so sessions survive restarts with no extra services."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    current_step INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    time_hours REAL NOT NULL,
                    request TEXT NOT NULL,
                    report TEXT NOT NULL,
                    particles BLOB NOT NULL,
                    PRIMARY KEY (session_id, step)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def create(self, session_id: str, model_id: str, spec: dict, report: dict, p: Particles) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?)",
                (session_id, time.time(), model_id, json.dumps(spec), 0),
            )
            db.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?)",
                (session_id, 0, 0.0, "{}", json.dumps(report), _encode(p)),
            )

    def meta(self, session_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT created_at, model_id, spec, current_step FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise SessionNotFound(session_id)
        return {"created_at": row[0], "model_id": row[1], "spec": json.loads(row[2]), "current_step": row[3]}

    def snapshot(self, session_id: str, step: int) -> tuple[float, dict, dict, Particles]:
        with self._connect() as db:
            row = db.execute(
                "SELECT time_hours, request, report, particles FROM snapshots WHERE session_id=? AND step=?",
                (session_id, step),
            ).fetchone()
        if row is None:
            raise StepNotFound(step)
        return row[0], json.loads(row[1]), json.loads(row[2]), _decode(row[3])

    def append(self, session_id: str, step: int, time_hours: float, request: dict, report: dict, p: Particles) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?)",
                (session_id, step, time_hours, json.dumps(request), json.dumps(report), _encode(p)),
            )
            db.execute("UPDATE sessions SET current_step=? WHERE id=?", (step, session_id))

    def rewind(self, session_id: str, to_step: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM snapshots WHERE session_id=? AND step>?", (session_id, to_step))
            db.execute("UPDATE sessions SET current_step=? WHERE id=?", (to_step, session_id))

    def replace(self, session_id: str, model_id: str, spec: dict, report: dict, p: Particles) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM snapshots WHERE session_id=?", (session_id,))
            db.execute(
                "UPDATE sessions SET model_id=?, spec=?, current_step=0 WHERE id=?",
                (model_id, json.dumps(spec), session_id),
            )
            db.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?)",
                (session_id, 0, 0.0, "{}", json.dumps(report), _encode(p)),
            )

    def fork(self, session_id: str, new_id: str, from_step: int) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions SELECT ?, ?, model_id, spec, ? FROM sessions WHERE id=?",
                (new_id, time.time(), from_step, session_id),
            )
            db.execute(
                """INSERT INTO snapshots SELECT ?, step, time_hours, request, report, particles
                   FROM snapshots WHERE session_id=? AND step<=?""",
                (new_id, session_id, from_step),
            )

    def history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT step, time_hours, request, report FROM snapshots WHERE session_id=? ORDER BY step",
                (session_id,),
            ).fetchall()
        return [{"step": r[0], "time_hours": r[1], "request": json.loads(r[2]), "result": json.loads(r[3])} for r in rows]

    def delete(self, session_id: str) -> None:
        self.meta(session_id)
        with self._connect() as db:
            db.execute("DELETE FROM snapshots WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def list_ids(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id, created_at, current_step FROM sessions ORDER BY created_at DESC").fetchall()
        return [{"session_id": r[0], "created_at": r[1], "current_step": r[2]} for r in rows]


class SessionManager:
    """Connects the simulator to session memory, one step at a time."""

    def __init__(self, simulator: Simulator, store: SessionStore):
        self.sim = simulator
        self.store = store
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    # ---- What the caller should know alongside every result ----

    def _model_block(self) -> dict[str, Any]:
        status = self.sim.meta.get("status", "prior_only")
        notes = {
            "prior_only": "Untrained. Results come from textbook equations and assumptions, not from fitted data.",
            "trained": "Trained on the data listed by GET /v1/model. Still a research simulation, not a medical tool.",
        }
        return {"id": self.sim.model_id, "status": status, "note": notes.get(status, "")}

    def _support(self, p: Particles, spec: dict, book: DrugBook, time_doses: list) -> dict[str, Any]:
        support = self.sim.meta.get("support")
        if not support:
            return {"within_training_support": False, "reasons": ["The model has not been trained on data yet."]}
        reasons = []
        for name, (low, high) in support.get("covariates", {}).items():
            column = p.cov[:, COV_INDEX[name]]
            if float(np.percentile(column, 5)) < low or float(np.percentile(column, 95)) > high:
                reasons.append(f"{name} goes outside the trained range {low:.1f} to {high:.1f}.")
        for cond in spec.get("conditions", []):
            if cond["name"] not in support.get("conditions", []):
                reasons.append(f"Condition {cond['name']} has no training data.")
        trained_drugs = support.get("drugs", {})
        for _, drug, mg in time_doses:
            if drug not in trained_drugs:
                reasons.append(f"Drug {drug} has no training data.")
            elif not trained_drugs[drug][0] * 0.5 <= mg <= trained_drugs[drug][1] * 1.5:
                reasons.append(f"Drug {drug} at {mg:g} mg is far from trained doses.")
        return {"within_training_support": not reasons, "reasons": sorted(set(reasons))}

    def _dose_log(self, spec: dict, session_id: str, upto_step: int) -> list[tuple[float, str, float]]:
        log = []
        for row in self.store.history(session_id):
            if row["step"] == 0 or row["step"] > upto_step:
                continue
            start = row["time_hours"] - row["request"].get("duration_hours", 0.0)
            for event in row["request"].get("events", []):
                if event["type"] == "dose":
                    log.append((start + event["at_hours"], event["drug"], float(event["dose_mg"])))
        return log

    def _warnings(self, spec: dict, book: DrugBook, log: list, start: float, events: list[dict]) -> list[str]:
        warnings = []
        allergies = {a.lower() for a in spec["patient"].get("allergies", [])}
        custom = {d["name"] for d in spec.get("custom_drugs", [])}
        for event in events:
            if event["type"] != "dose":
                continue
            drug, mg, at = event["drug"], float(event["dose_mg"]), start + event["at_hours"]
            if drug.lower() in allergies:
                warnings.append(f"The patient lists an allergy to {drug}. This version does not simulate allergic reactions.")
            if drug in custom:
                warnings.append(f"{drug} is hypothetical. Its effects come only from the parameters supplied.")
            if event.get("route") == "iv" and drug in KNOWLEDGE.drug_names:
                if float(KNOWLEDGE.drug_meta(drug)["bioavailability"]) == 1.0:
                    warnings.append(f"{drug} parameters are apparent oral values, so IV results are unreliable.")
            limits = book.limits.get(drug)
            if limits:
                if mg > limits["max_single_dose_mg"]:
                    warnings.append(f"{drug} {mg:g} mg exceeds the single-dose warning level of {limits['max_single_dose_mg']:g} mg.")
                daily = sum(m for t, d, m in log if d == drug and at - 24.0 < t <= at)
                if daily > limits["max_daily_dose_mg"]:
                    warnings.append(f"{drug} reaches {daily:g} mg in 24 h, above the warning level of {limits['max_daily_dose_mg']:g} mg.")
        return sorted(set(warnings))

    def _result(self, session_id, step, time_hours, p, spec, book, log, include_internal, warnings):
        report = self.sim.report(p, book, include_internal)
        return {
            "session_id": session_id,
            "step": step,
            "time_hours": round(time_hours, 4),
            **report,
            "warnings": warnings,
            "support": self._support(p, spec, book, log),
            "model": self._model_block(),
        }

    # ---- Session lifecycle ----

    def create(self, spec: dict[str, Any], people: int, seed: int | None, include_internal: bool = False) -> dict[str, Any]:
        seed = int(secrets.randbits(31)) if seed is None else int(seed)
        book = DrugBook.build(spec.get("custom_drugs"))
        p, imputed = self.sim.create_particles(
            {k: v for k, v in spec["patient"].items() if k != "allergies"}, spec.get("conditions", []), book, people, seed
        )
        session_id = "s_" + secrets.token_hex(8)
        full_spec = {**spec, "seed": seed, "people": people, "imputed": imputed, "baseline": patient_summary(p)}
        result = self._result(session_id, 0, 0.0, p, full_spec, book, [], include_internal, [])
        result.update({"seed": seed, "imputed_parameters": imputed})
        self.store.create(session_id, self.sim.model_id, full_spec, result, p)
        return result

    def _check_model(self, meta: dict, session_id: str) -> None:
        if meta["model_id"] != self.sim.model_id:
            raise ModelMismatch(
                f"Session {session_id} was created with model {meta['model_id']}, but the server now runs "
                f"{self.sim.model_id}. Reset it with new_people true, or create a new session."
            )

    def step(self, session_id: str, duration: Any, events: list[dict], include_internal: bool = False) -> dict[str, Any]:
        with self._lock(session_id):
            meta = self.store.meta(session_id)
            self._check_model(meta, session_id)
            spec = meta["spec"]
            duration_h = parse_duration(duration)
            expanded = expand_events(events, duration_h)
            if len(expanded) > MAX_EVENTS_PER_STEP:
                raise PatientError(f"One step can hold at most {MAX_EVENTS_PER_STEP} events.")
            book = DrugBook.build(spec.get("custom_drugs"))
            current = meta["current_step"]
            start, _, _, p = self.store.snapshot(session_id, current)

            log = self._dose_log(spec, session_id, current)
            log += [(start + e["at_hours"], e["drug"], float(e["dose_mg"])) for e in expanded if e["type"] == "dose"]
            warnings = self._warnings(spec, book, log, start, expanded)

            p_next = self.sim.advance(p, book, duration_h, expanded)
            time_hours = start + duration_h
            result = self._result(session_id, current + 1, time_hours, p_next, spec, book, log, include_internal, warnings)
            result["applied_events"] = expanded
            request = {"duration_hours": duration_h, "events": expanded}
            self.store.append(session_id, current + 1, time_hours, request, result, p_next)
            return result

    def current(self, session_id: str, include_internal: bool = False) -> dict[str, Any]:
        meta = self.store.meta(session_id)
        self._check_model(meta, session_id)
        spec = meta["spec"]
        book = DrugBook.build(spec.get("custom_drugs"))
        time_hours, _, _, p = self.store.snapshot(session_id, meta["current_step"])
        log = self._dose_log(spec, session_id, meta["current_step"])
        result = self._result(session_id, meta["current_step"], time_hours, p, spec, book, log, include_internal, [])
        result.update(
            {
                "patient": spec["patient"],
                "conditions": spec.get("conditions", []),
                "seed": spec["seed"],
                "imputed_parameters": spec["imputed"],
            }
        )
        return result

    def reset(
        self, session_id: str, to_step: int = 0, new_people: bool = False, seed: int | None = None
    ) -> dict[str, Any]:
        """Forgets everything after to_step. With new_people, also draws new virtual people."""
        with self._lock(session_id):
            meta = self.store.meta(session_id)
            if new_people:
                spec = {k: meta["spec"][k] for k in ("patient", "conditions", "custom_drugs") if k in meta["spec"]}
                seed = int(secrets.randbits(31)) if seed is None else int(seed)
                book = DrugBook.build(spec.get("custom_drugs"))
                people = int(meta["spec"]["people"])
                p, imputed = self.sim.create_particles(
                    {k: v for k, v in spec["patient"].items() if k != "allergies"},
                    spec.get("conditions", []),
                    book,
                    people,
                    seed,
                )
                full_spec = {**spec, "seed": seed, "people": people, "imputed": imputed, "baseline": patient_summary(p)}
                result = self._result(session_id, 0, 0.0, p, full_spec, book, [], False, [])
                result.update({"seed": seed, "imputed_parameters": imputed})
                self.store.replace(session_id, self.sim.model_id, full_spec, result, p)
                return result
            self._check_model(meta, session_id)
            if to_step < 0 or to_step > meta["current_step"]:
                raise StepNotFound(to_step)
            self.store.rewind(session_id, to_step)
        return self.current(session_id)

    def fork(self, session_id: str, from_step: int | None = None) -> dict[str, Any]:
        with self._lock(session_id):
            meta = self.store.meta(session_id)
            self._check_model(meta, session_id)
            step = meta["current_step"] if from_step is None else from_step
            if step < 0 or step > meta["current_step"]:
                raise StepNotFound(step)
            new_id = "s_" + secrets.token_hex(8)
            self.store.fork(session_id, new_id, step)
        result = self.current(new_id)
        result["forked_from"] = {"session_id": session_id, "step": step}
        return result
