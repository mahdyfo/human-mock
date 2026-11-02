"""Runs virtual people forward in time.

A session does not simulate one person. It simulates a cloud of plausible people
who all match the description, because a description never pins down one body.
The spread across that cloud is the uncertainty reported to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE, MAX_DRUGS, build_drug_row
from humanmock.model import (
    COV_INDEX,
    FRACTIONS,
    LAYOUT,
    N_LATENT,
    OBS_INDEX,
    OBS_NAMES,
    SERVE_SOLVER,
    BodyModel,
    SolverConfig,
    advance,
)
from humanmock.population import PatientError, Population

NO_EXPOSURE = -10.0
PUBLIC_OBSERVABLES: dict[str, str] = {
    "temperature_c": "degrees C",
    "heart_rate_bpm": "beats per minute",
    "respiratory_rate_bpm": "breaths per minute",
    "throat_pain_0_10": "score 0 to 10",
    "cough_0_10": "score 0 to 10",
    "fatigue_0_10": "score 0 to 10",
    "diarrhea_0_10": "score 0 to 10",
    "wbc_k_ul": "thousand cells per microlitre",
    "crp_mg_l": "mg per litre",
    "throat_culture_log10_cfu": "log10 colony forming units",
    "toxicity_index_0_1": "index 0 to 1",
}
INTERNAL_STATES = (
    "log_surface",
    "log_reservoir",
    "innate",
    "effector",
    "memory",
    "cytokine",
    "tissue",
    "dysbiosis",
    "toxicity",
)


@eqx.filter_jit
def _contexts(model: BodyModel, p: dict, table):
    return jax.vmap(lambda cov, es, ep, ed: model.context(cov, es, ep, ed, table))(
        p["cov"], p["eta_sys"], p["eta_path"], p["eta_drug"]
    )


@eqx.filter_jit
def _initial_states(model: BodyModel, p: dict, table):
    def one(cov, es, ep, ed):
        return model.initial_state(model.context(cov, es, ep, ed, table))

    return jax.vmap(one)(p["cov"], p["eta_sys"], p["eta_path"], p["eta_drug"])


@eqx.filter_jit
def _segment(model: BodyModel, y, p: dict, table, oral, iv, exposure, dt, cfg: SolverConfig):
    """Applies instantaneous events, then advances each virtual person by their own dt."""

    def one(y_i, cov, es, ep, ed, exposure_i, dt_i):
        c = model.context(cov, es, ep, ed, table)
        y_i = model.apply_events(y_i, c, oral, iv, exposure_i)
        return advance(model, y_i, c, dt_i, cfg)

    return jax.vmap(one)(y, p["cov"], p["eta_sys"], p["eta_path"], p["eta_drug"], exposure, dt)


@eqx.filter_jit
def _observe(model: BodyModel, y, p: dict, table):
    def one(y_i, cov, es, ep, ed):
        return model.observe(y_i, model.context(cov, es, ep, ed, table))

    return jax.vmap(one)(y, p["cov"], p["eta_sys"], p["eta_path"], p["eta_drug"])


@dataclass
class Particles:
    """The memory of a session. Saving this and restoring it resumes the same bodies."""

    y: np.ndarray
    cov: np.ndarray
    eta_sys: np.ndarray
    eta_path: np.ndarray
    eta_drug: np.ndarray
    member: np.ndarray

    @property
    def n(self) -> int:
        return self.y.shape[0]

    def params(self, mask: np.ndarray) -> dict:
        return {
            "cov": jnp.asarray(self.cov[mask]),
            "eta_sys": jnp.asarray(self.eta_sys[mask]),
            "eta_path": jnp.asarray(self.eta_path[mask]),
            "eta_drug": jnp.asarray(self.eta_drug[mask]),
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        return dict(self.__dict__)

    @staticmethod
    def from_arrays(arrays: dict[str, np.ndarray]) -> "Particles":
        return Particles(**{k: arrays[k] for k in ("y", "cov", "eta_sys", "eta_path", "eta_drug", "member")})


@dataclass
class DrugBook:
    """Which drug sits in which slot for a session, including hypothetical drugs."""

    names: list[str]
    table: np.ndarray
    limits: dict[str, dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def build(custom: list[dict[str, Any]] | None = None) -> "DrugBook":
        names = list(KNOWLEDGE.drug_names)
        table = KNOWLEDGE.drug_table.copy()
        limits = {
            n: {
                "max_single_dose_mg": float(KNOWLEDGE.drug_meta(n).get("max_single_dose_mg", np.inf)),
                "max_daily_dose_mg": float(KNOWLEDGE.drug_meta(n).get("max_daily_dose_mg", np.inf)),
            }
            for n in names
        }
        for spec in custom or []:
            name = spec["name"]
            if name in names:
                raise PatientError(f"Drug '{name}' already exists.")
            if len(names) >= MAX_DRUGS:
                raise PatientError(f"A session can hold at most {MAX_DRUGS} drugs.")
            table[len(names)] = build_drug_row(spec)
            names.append(name)
        return DrugBook(names=names, table=table, limits=limits)

    def slot(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise PatientError(f"Unknown drug '{name}'. Known drugs are {', '.join(self.names)}.") from None


def expand_events(events: list[dict[str, Any]], duration_h: float) -> list[dict[str, Any]]:
    """Turns repeating doses into individual events inside this step."""
    out = []
    for event in events:
        repeat = int(event.get("repeat") or 1)
        every = float(event.get("every_hours") or 0.0)
        if repeat > 1 and every <= 0:
            raise PatientError("An event with repeat above 1 needs every_hours above 0.")
        for k in range(repeat):
            at = float(event.get("at_hours", 0.0)) + k * every
            if at < 0 or at >= duration_h:
                raise PatientError(
                    f"Event at {at} h is outside this step of {duration_h} h. Lengthen duration or move the event."
                )
            item = {key: v for key, v in event.items() if key not in ("repeat", "every_hours")}
            item["at_hours"] = at
            out.append(item)
    return sorted(out, key=lambda e: e["at_hours"])


class Simulator:
    def __init__(
        self,
        members: list[BodyModel] | None = None,
        meta: dict[str, Any] | None = None,
        population: Population | None = None,
        solver: SolverConfig = SERVE_SOLVER,
    ):
        self.members = members or [BodyModel.create(jax.random.PRNGKey(0))]
        self.meta = meta or {"status": "prior_only", "model_id": "prior-v0"}
        self.population = population or Population()
        self.solver = solver

    @property
    def model_id(self) -> str:
        return str(self.meta.get("model_id", "prior-v0"))

    # ---- Creating people ----

    def create_particles(
        self,
        patient: dict[str, Any],
        conditions: list[dict[str, Any]],
        book: DrugBook,
        n: int,
        seed: int,
    ) -> tuple[Particles, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        cov, imputed = self.population.sample(patient, n, rng)
        p = Particles(
            y=np.zeros((n, LAYOUT.size), dtype=np.float32),
            cov=cov.astype(np.float32),
            eta_sys=rng.standard_normal((n, len(KNOWLEDGE.sys_names))).astype(np.float32),
            eta_path=rng.standard_normal((n, KNOWLEDGE.n_pathogens, len(KNOWLEDGE.path_fields))).astype(np.float32),
            eta_drug=rng.standard_normal((n, MAX_DRUGS, len(DRUG_FIELDS))).astype(np.float32),
            member=np.arange(n) % len(self.members),
        )
        table = jnp.asarray(book.table)
        for m, model in enumerate(self.members):
            mask = p.member == m
            p.y[mask] = np.asarray(_initial_states(model, p.params(mask), table))
        self._burn_in(p, conditions, table)
        return p, imputed

    def _burn_in(self, p: Particles, conditions: list[dict[str, Any]], table) -> None:
        """Simulates from exposure up to now, so an illness starts already under way."""
        if not conditions:
            return
        path_i = {f: i for i, f in enumerate(KNOWLEDGE.path_fields)}
        for m, model in enumerate(self.members):
            mask = p.member == m
            params = p.params(mask)
            ctx = _contexts(model, params, table)
            plans = []
            for cond in conditions:
                if cond["name"] not in KNOWLEDGE.path_names:
                    raise PatientError(
                        f"Unknown condition '{cond['name']}'. Known conditions are {', '.join(KNOWLEDGE.path_names)}."
                    )
                slot = KNOWLEDGE.path_names.index(cond["name"])
                onset_h = float(cond.get("days_since_symptom_onset", 1.0)) * 24.0
                before = np.asarray(ctx.path["incubation_h"][:, slot]) + onset_h
                inoculum = np.asarray(ctx.path["log10_inoculum"][:, slot])
                plans.append((slot, before, inoculum))
            plans.sort(key=lambda plan: -plan[1].mean())
            y = jnp.asarray(p.y[mask])
            zero = jnp.zeros(MAX_DRUGS)
            for idx, (slot, before, inoculum) in enumerate(plans):
                following = plans[idx + 1][1] if idx + 1 < len(plans) else np.zeros_like(before)
                dt = np.maximum(before - following, 0.0).astype(np.float32)
                exposure = np.full((len(dt), KNOWLEDGE.n_pathogens), NO_EXPOSURE, dtype=np.float32)
                exposure[:, slot] = inoculum
                y = _segment(model, y, params, table, zero, zero, jnp.asarray(exposure), jnp.asarray(dt), self.solver)
            p.y[mask] = np.asarray(y)
        del path_i

    # ---- Moving through time ----

    def advance(
        self, p: Particles, book: DrugBook, duration_h: float, events: list[dict[str, Any]]
    ) -> Particles:
        """Advances every virtual person, stopping at each event. Returns new particles."""
        events = expand_events(events, duration_h)
        table = jnp.asarray(book.table)
        y = p.y.copy()
        cuts = sorted({e["at_hours"] for e in events} | {0.0})
        cuts.append(duration_h)
        for start, end in zip(cuts[:-1], cuts[1:]):
            oral = np.zeros(MAX_DRUGS, dtype=np.float32)
            iv = np.zeros(MAX_DRUGS, dtype=np.float32)
            exposure = np.full(KNOWLEDGE.n_pathogens, NO_EXPOSURE, dtype=np.float32)
            for event in (e for e in events if e["at_hours"] == start):
                if event["type"] == "dose":
                    slot = book.slot(event["drug"])
                    if event.get("route", "oral") == "iv":
                        iv[slot] += float(event["dose_mg"])
                    else:
                        oral[slot] += float(event["dose_mg"])
                elif event["type"] == "exposure":
                    if event["pathogen"] not in KNOWLEDGE.path_names:
                        raise PatientError(f"Unknown pathogen '{event['pathogen']}'.")
                    slot = KNOWLEDGE.path_names.index(event["pathogen"])
                    field_i = KNOWLEDGE.path_fields.index("log10_inoculum")
                    default = float(KNOWLEDGE.path_values[slot, field_i])
                    exposure[slot] = float(event.get("log10_inoculum", default))
                else:
                    raise PatientError(f"Unknown event type '{event['type']}'.")
            for m, model in enumerate(self.members):
                mask = p.member == m
                count = int(mask.sum())
                if count == 0:
                    continue
                y[mask] = np.asarray(
                    _segment(
                        model,
                        jnp.asarray(y[mask]),
                        p.params(mask),
                        table,
                        jnp.asarray(oral),
                        jnp.asarray(iv),
                        jnp.tile(jnp.asarray(exposure), (count, 1)),
                        jnp.full(count, float(end - start), dtype=jnp.float32),
                        self.solver,
                    )
                )
        return Particles(y=y, cov=p.cov, eta_sys=p.eta_sys, eta_path=p.eta_path, eta_drug=p.eta_drug, member=p.member)

    # ---- Reading results ----

    def observe(self, p: Particles, book: DrugBook) -> np.ndarray:
        table = jnp.asarray(book.table)
        out = np.zeros((p.n, len(OBS_NAMES)), dtype=np.float32)
        for m, model in enumerate(self.members):
            mask = p.member == m
            if mask.any():
                out[mask] = np.asarray(_observe(model, jnp.asarray(p.y[mask]), p.params(mask), table))
        return out

    def report(self, p: Particles, book: DrugBook, include_internal: bool = False) -> dict[str, Any]:
        obs = self.observe(p, book)
        healthy = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(p.y), axis=1)
        result: dict[str, Any] = {"virtual_people": int(p.n), "failed_virtual_people": int((~healthy).sum())}
        if not healthy.any():
            raise RuntimeError("Every virtual person failed to integrate.")
        good = obs[healthy]

        def spread(column: np.ndarray) -> dict[str, float]:
            low, mid, high = np.percentile(column, [10, 50, 90])
            return {"p10": round(float(low), 3), "p50": round(float(mid), 3), "p90": round(float(high), 3)}

        result["observations"] = {name: spread(good[:, OBS_INDEX[name]]) for name in PUBLIC_OBSERVABLES}
        levels = {}
        for slot, name in enumerate(book.names):
            column = good[:, OBS_INDEX[f"conc_slot{slot}"]]
            if float(np.percentile(column, 90)) > 1e-3:
                levels[name] = spread(column)
        result["drug_levels_mg_l"] = levels
        result["share_of_people"] = {
            name.replace("_fraction", ""): round(float((good[:, OBS_INDEX[src]] > thr).mean()), 3)
            for name, (src, thr, _) in FRACTIONS.items()
        }
        if include_internal:
            internal: dict[str, Any] = {}
            y = p.y[healthy]
            for name in INTERNAL_STATES:
                _, off, size, _, _ = LAYOUT.by_name[name]
                if size == 1:
                    internal[name] = spread(y[:, off])
                else:
                    labels = KNOWLEDGE.path_names if size == KNOWLEDGE.n_pathogens else book.names
                    internal[name] = {labels[j]: spread(y[:, off + j]) for j in range(size)}
            latents = y[:, LAYOUT.physio_size :]
            internal["memory_neuron_activity"] = spread(np.abs(latents).max(axis=1))
            internal["memory_neurons_per_organ"] = N_LATENT
            result["internal"] = internal
        return result


def patient_summary(p: Particles) -> dict[str, float]:
    return {name: round(float(p.cov[:, i].mean()), 3) for name, i in COV_INDEX.items()}
