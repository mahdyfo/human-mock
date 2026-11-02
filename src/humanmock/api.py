"""The HTTP API.

Create a session, step it forward with doses and exposures, read the body back,
and reset or fork it. Start it with: humanmock serve
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from humanmock import __version__
from humanmock.engine import Simulator
from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE
from humanmock.model import PUBLIC_OBSERVABLES, trainable_groups
from humanmock.population import PatientError
from humanmock.sessions import ModelMismatch, SessionManager, SessionNotFound, SessionStore, StepNotFound


class Strict(BaseModel):
    # An unknown field is rejected, so a misspelled parameter never silently becomes a default.
    model_config = ConfigDict(extra="forbid")


def _patient_fields() -> dict[str, Any]:
    """Patient parameters come from physiology.yaml, so adding one there adds it here."""
    fields: dict[str, Any] = {}
    for name, entry in KNOWLEDGE.patient.items():
        if name == "sex_female":
            fields["sex"] = (Literal["male", "female"] | None, Field(None))
        else:
            fields[name] = (float | None, Field(None, description=entry["note"]))
    fields["allergies"] = (list[str], Field(default_factory=list))
    return fields


Patient = create_model(
    "Patient",
    __base__=Strict,
    __doc__="Every field is optional. Anything left out is sampled from similar people.",
    **_patient_fields(),
)


class Condition(Strict):
    name: str
    days_since_symptom_onset: float = Field(1.0, ge=0, le=30)


class CustomDrug(Strict):
    """A hypothetical drug described by mechanism rather than by name."""

    name: str = Field(min_length=1, max_length=40)
    ka: float = Field(1.0, gt=0, le=20, description="Absorption rate per hour.")
    bioavailability: float = Field(1.0, gt=0, le=1)
    v_l_per_kg: float = Field(1.0, gt=0, le=50)
    cl_ref_l_h: float = Field(10.0, gt=0, le=500, description="Clearance at 70 kg in L per hour.")
    frac_renal: float = Field(0.5, ge=0, le=1)
    frac_unbound: float = Field(1.0, gt=0, le=1)
    kill_emax: float = Field(0.0, ge=0, le=20, description="Largest antibacterial kill rate per hour.")
    kill_ec50: float = Field(1.0, gt=0)
    kill_hill: float = Field(1.0, gt=0, le=8)
    reservoir_penetration: float = Field(0.0, ge=0, le=1)
    antipyretic_emax: float = Field(0.0, ge=0, le=1)
    antipyretic_ec50: float = Field(1.0, gt=0)
    analgesic_emax: float = Field(0.0, ge=0, le=1)
    analgesic_ec50: float = Field(1.0, gt=0)
    gut_rate: float = Field(0.0, ge=0, le=1)
    gut_ec50: float = Field(1.0, gt=0)
    tox_rate: float = Field(0.0, ge=0, le=5)
    tox_ec50: float = Field(1.0, gt=0)


class CreateSession(Strict):
    patient: Patient = Field(default_factory=Patient)
    conditions: list[Condition] = Field(default_factory=list)
    custom_drugs: list[CustomDrug] = Field(default_factory=list)
    people: int = Field(64, ge=1, le=512, description="How many plausible virtual people to simulate.")
    seed: int | None = None
    include_internal: bool = False


class Event(Strict):
    type: Literal["dose", "exposure"] = "dose"
    at_hours: float = Field(0.0, ge=0)
    drug: str | None = None
    dose_mg: float | None = Field(None, gt=0, le=200000)
    route: Literal["oral", "iv"] = "oral"
    pathogen: str | None = None
    log10_inoculum: float | None = Field(None, ge=0, le=9)
    repeat: int = Field(1, ge=1, le=500)
    every_hours: float | None = Field(None, gt=0)


class Step(Strict):
    duration: str | float = Field(description="A number of hours, or text such as 30m, 8h, or 3d.")
    events: list[Event] = Field(default_factory=list)
    include_internal: bool = False


class Reset(Strict):
    to_step: int = Field(0, ge=0, description="Return to this step and forget everything after it.")
    new_people: bool = Field(False, description="Also draw new virtual people for the same description.")
    seed: int | None = None


class Fork(Strict):
    from_step: int | None = Field(None, ge=0)


def _event_dicts(events: list[Event]) -> list[dict[str, Any]]:
    out = []
    for event in events:
        if event.type == "dose" and (not event.drug or event.dose_mg is None):
            raise PatientError("A dose event needs both drug and dose_mg.")
        if event.type == "exposure" and not event.pathogen:
            raise PatientError("An exposure event needs a pathogen.")
        out.append(event.model_dump(exclude_none=True))
    return out


def create_app(simulator: Simulator | None = None, database: str | Path | None = None) -> FastAPI:
    if simulator is None:
        checkpoint = os.environ.get("HUMANMOCK_CHECKPOINT")
        if checkpoint:
            from humanmock.checkpoint import load_simulator

            simulator = load_simulator(checkpoint)
        else:
            simulator = Simulator()
    database = database or os.environ.get("HUMANMOCK_DB", "var/sessions.db")
    manager = SessionManager(simulator, SessionStore(database))

    app = FastAPI(
        title="humanmock",
        version=__version__,
        description="An open research simulator of human body behaviour. It is not a medical tool.",
    )

    def guarded(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except SessionNotFound:
            raise HTTPException(404, "Session not found.") from None
        except StepNotFound as exc:
            raise HTTPException(404, f"Step {exc.args[0]} does not exist in this session.") from None
        except ModelMismatch as exc:
            raise HTTPException(409, str(exc)) from None
        except PatientError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/healthz")
    def health():
        return {"status": "ok", "model": simulator.model_id}

    @app.get("/v1/catalog")
    def catalog():
        """Lists what can be simulated, for people and for agents discovering the API."""
        drugs = {
            n: {
                "label": KNOWLEDGE.drug_meta(n)["label"],
                "max_single_dose_mg": KNOWLEDGE.drug_meta(n).get("max_single_dose_mg"),
                "max_daily_dose_mg": KNOWLEDGE.drug_meta(n).get("max_daily_dose_mg"),
                "sources": KNOWLEDGE.drug_meta(n).get("sources", {}),
            }
            for n in KNOWLEDGE.drug_names
        }
        conditions = {n: KNOWLEDGE.raw["pathogens"][n]["label"] for n in KNOWLEDGE.path_names}
        return {
            "drugs": drugs,
            "conditions": conditions,
            "observables": PUBLIC_OBSERVABLES,
            "patient_parameters": [f for f in Patient.model_fields if f != "allergies"],
            "custom_drug_parameters": list(DRUG_FIELDS),
        }

    @app.get("/v1/model")
    def model():
        return {
            "id": simulator.model_id,
            "status": simulator.meta.get("status"),
            "ensemble_size": len(simulator.members),
            "trainable_parameters": trainable_groups(simulator.members[0]),
            "details": {k: v for k, v in simulator.meta.items() if k not in ("status", "model_id")},
        }

    @app.post("/v1/sessions", status_code=201)
    def create_session(body: CreateSession):
        spec = {
            "patient": body.patient.model_dump(exclude_none=True),
            "conditions": [c.model_dump() for c in body.conditions],
            "custom_drugs": [d.model_dump() for d in body.custom_drugs],
        }
        return guarded(manager.create, spec, body.people, body.seed, body.include_internal)

    @app.get("/v1/sessions")
    def list_sessions():
        return {"sessions": manager.store.list_ids()}

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str, include_internal: bool = False):
        return guarded(manager.current, session_id, include_internal)

    @app.post("/v1/sessions/{session_id}/step")
    def step_session(session_id: str, body: Step):
        return guarded(manager.step, session_id, body.duration, guarded(_event_dicts, body.events), body.include_internal)

    @app.post("/v1/sessions/{session_id}/reset")
    def reset_session(session_id: str, body: Reset | None = None):
        body = body or Reset()
        return guarded(manager.reset, session_id, body.to_step, body.new_people, body.seed)

    @app.post("/v1/sessions/{session_id}/fork", status_code=201)
    def fork_session(session_id: str, body: Fork | None = None):
        body = body or Fork()
        return guarded(manager.fork, session_id, body.from_step)

    @app.get("/v1/sessions/{session_id}/history")
    def session_history(session_id: str):
        guarded(manager.store.meta, session_id)
        return {"session_id": session_id, "steps": manager.store.history(session_id)}

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        guarded(manager.store.delete, session_id)

    return app
