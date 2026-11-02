"""The trainable body: known equations plus neural cells with memory neurons.

The state of one virtual person is a single vector with two parts.

Physiological states are named quantities such as drug level or temperature.
Their rate of change is the known biology from the organ modules plus a neural
correction that training can grow.

Memory neurons are leaky integrators attached to each organ. Each organ owns
four of them, running from hours to months. They let the network remember
things the equations have no variable for, such as tolerance or slow adaptation,
and they persist for as long as a session does.

Every neural output starts at exactly zero, so an untrained model behaves as
pure textbook physiology. Training can only move it away from that where data
gives a reason, and the correction gates below stop it breaking basic physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE, MAX_DRUGS
from humanmock.organs import ORGANS
from humanmock.organs.base import AMOUNT, COUNT, FRACTION, KINDS, LEVEL, PER_DRUG, PER_PATHOGEN, sigmoid

COVARIATES: tuple[str, ...] = tuple(KNOWLEDGE.patient)
COV_INDEX = {n: i for i, n in enumerate(COVARIATES)}
COV_CENTER = np.array([KNOWLEDGE.patient[n]["center"] for n in COVARIATES], dtype=np.float32)
COV_SCALE = np.array([KNOWLEDGE.patient[n]["scale"] for n in COVARIATES], dtype=np.float32)

N_LATENT = 4
LATENT_TAUS_H = np.array([6.0, 24.0, 168.0, 1440.0], dtype=np.float32)
EMBED_SIZE = 8
HIDDEN = 32


class OrganError(ValueError):
    """Raised when an organ's declaration does not fit the body."""


def check_organs(organs) -> list[str]:
    """Returns every mistake in the organ declarations, in plain language."""
    problems: list[str] = []
    names = [o.name for o in organs]
    seen_states: set[str] = set()
    seen_measurements: set[str] = set()

    def unique(kind: str, name: str, seen: set[str]) -> None:
        if name in seen:
            problems.append(f"Two organs declare a {kind} named {name}.")
        seen.add(name)

    if len(names) != len(set(names)):
        problems.append("Two organs have the same name.")
    for organ in organs:
        for neighbor in organ.neighbors:
            if neighbor not in names:
                problems.append(f"Organ {organ.name} lists unknown neighbor {neighbor}. Known organs are {names}.")
        for st in organ.states:
            unique("state", st.name, seen_states)
            if st.kind not in KINDS:
                problems.append(f"State {st.name} has unknown kind {st.kind}. Use one of {KINDS}.")
            if st.per not in (None, PER_DRUG, PER_PATHOGEN):
                problems.append(f"State {st.name} has unknown per {st.per}. Use PER_DRUG, PER_PATHOGEN, or leave it out.")
            if st.init_from is not None and st.init_from not in COVARIATES:
                problems.append(f"State {st.name} starts from unknown patient parameter {st.init_from}.")
        if organ.measurements and organ.observables is None:
            problems.append(f"Organ {organ.name} declares measurements but has no observables function.")
        for measurement in organ.measurements:
            unique("measurement", measurement.name, seen_measurements)
        for share in organ.shares:
            unique("measurement", share.name, seen_measurements)
    every_measurement = {m.name for o in organs for m in o.measurements}
    for organ in organs:
        for share in organ.shares:
            if share.measurement not in every_measurement:
                problems.append(f"Share {share.name} reads unknown measurement {share.measurement}.")
    return problems


_problems = check_organs(ORGANS)
if _problems:
    raise OrganError("The organ declarations have problems:\n- " + "\n- ".join(_problems))


class Layout:
    """Maps between named organ states and the single vector the solver integrates."""

    def __init__(self):
        self.entries = []
        offset = 0
        for organ in ORGANS:
            for st in organ.states:
                self.entries.append((st.name, offset, st.size, st, organ.name))
                offset += st.size
        self.physio_size = offset
        self.latent_offset = {}
        for organ in ORGANS:
            self.latent_offset[organ.name] = offset
            offset += N_LATENT
        self.size = offset
        self.by_name = {e[0]: e for e in self.entries}
        self.organ_entries = {o.name: [e for e in self.entries if e[4] == o.name] for o in ORGANS}

    def unpack(self, y):
        s = {}
        for name, off, size, st, _ in self.entries:
            s[name] = y[off] if size == 1 else y[off : off + size]
        return s

    def pack_derivative(self, d, latents):
        parts = []
        for name, _, size, _, _ in self.entries:
            parts.append(jnp.atleast_1d(d[name]))
        parts.extend(latents)
        return jnp.concatenate(parts)

    def get(self, y, name):
        _, off, size, _, _ = self.by_name[name]
        return y[..., off] if size == 1 else y[..., off : off + size]

    def set(self, y, name, value):
        _, off, size, _, _ = self.by_name[name]
        return y.at[off].set(value) if size == 1 else y.at[off : off + size].set(value)


LAYOUT = Layout()

MEASUREMENTS = tuple(m for o in ORGANS for m in o.measurements)
SHARES = tuple(s for o in ORGANS for s in o.shares)
OBS_BASE: tuple[str, ...] = tuple(m.name for m in MEASUREMENTS)
OBS_NAMES: tuple[str, ...] = OBS_BASE + tuple(s.name for s in SHARES)
OBS_INDEX = {n: i for i, n in enumerate(OBS_NAMES)}
PUBLIC_OBSERVABLES: dict[str, str] = {m.name: m.unit for m in MEASUREMENTS if m.public}
_RESIDUAL_SCALE = np.array([m.neural_range for m in MEASUREMENTS], dtype=np.float32)
_OBS_SIGMA = np.array([m.noise for m in MEASUREMENTS] + [s.noise for s in SHARES], dtype=np.float32)

_DRUG_FIELD_INDEX = {n: i for i, n in enumerate(DRUG_FIELDS)}
_REGISTRY_MASK = np.array([i < len(KNOWLEDGE.drug_names) for i in range(MAX_DRUGS)], dtype=np.float32)


class Context(NamedTuple):
    """Everything fixed about one virtual person during a stretch of simulation."""

    sys: dict
    path: dict
    drug: dict
    patient: dict
    embed: jax.Array
    slot_trained: jax.Array


class _Signals(dict):
    """Values organs publish for each other, with a clear message for a wrong order."""

    def __missing__(self, name):
        raise KeyError(
            f"No organ has published the signal '{name}' yet. An organ's signals can only read signals from "
            f"organs above it in organs/__init__.py. Published so far: {sorted(self)}."
        )


def describe_body() -> list[dict]:
    """The body's own declarations, so anything added to it can appear in a UI by itself."""
    return [
        {
            "name": organ.name,
            "description": organ.description,
            "measurements": [
                {
                    "name": m.name,
                    "label": m.title,
                    "unit": m.unit,
                    "noise": m.noise,
                    "body_part": m.body_part,
                    "healthy": m.healthy,
                    "severe": m.severe,
                    "germs": m.germs,
                }
                for m in organ.measurements
                if m.public
            ],
            "states": [{"name": st.name, "label": st.title, "per": st.per} for st in organ.states if st.internal],
            "shares": [
                {"key": s.key, "label": s.title, "measurement": s.measurement, "threshold": s.threshold}
                for s in organ.shares
            ],
        }
        for organ in ORGANS
    ]


def _publish_signals(s, c: Context) -> _Signals:
    e = _Signals()
    for organ in ORGANS:
        if organ.signals is not None:
            organ.signals(s, c, e)
    return e


def _state_features(x, st):
    if st.kind == AMOUNT:
        return jnp.log1p(jnp.maximum(x, 0.0) / st.scale)
    if st.kind == FRACTION:
        return x
    return (x - st.center) / st.scale


def _gate(raw, x, st):
    """Applies a bounded neural correction in a way that cannot violate basic physics."""
    rate = st.rate_scale * jnp.tanh(raw)
    if st.kind == AMOUNT:
        return rate * jnp.maximum(x, 0.0)  # Proportional, so nothing appears from zero.
    if st.kind == COUNT:
        return rate * sigmoid((x - 1.0) / 0.15)  # An extinct population stays extinct.
    if st.kind == FRACTION:
        return jnp.where(rate > 0, rate * (1.0 - x), rate * jnp.maximum(x, 0.0))
    if st.kind == LEVEL:
        floor = jnp.maximum(x, 0.0) / (jnp.maximum(x, 0.0) + 0.1)
        return jnp.where(rate > 0, rate, rate * floor)
    return rate


def _zero_last_layer(mlp: eqx.nn.MLP) -> eqx.nn.MLP:
    last = mlp.layers[-1]
    mlp = eqx.tree_at(lambda m: m.layers[-1].weight, mlp, jnp.zeros_like(last.weight))
    return eqx.tree_at(lambda m: m.layers[-1].bias, mlp, jnp.zeros_like(last.bias))


def _cell_input_size(organ) -> int:
    own = sum(st.size for st in organ.states)
    others = 0
    for name in organ.neighbors:
        other = next(o for o in ORGANS if o.name == name)
        others += sum(st.size for st in other.states)
    return own + others + N_LATENT + EMBED_SIZE


class BodyModel(eqx.Module):
    log_sys: jax.Array
    log_path: jax.Array
    log_drug: jax.Array
    log_iiv: jax.Array
    log_obs_sigma: jax.Array
    log_tau: jax.Array
    encoder: eqx.nn.MLP
    cells: tuple
    decoder: eqx.nn.MLP

    @staticmethod
    def create(key) -> "BodyModel":
        keys = jax.random.split(key, len(ORGANS) + 2)
        n_sys = len(KNOWLEDGE.sys_names)
        n_path = len(KNOWLEDGE.path_fields)
        n_drug = len(DRUG_FIELDS)
        encoder = eqx.nn.MLP(
            len(COVARIATES),
            EMBED_SIZE + n_sys + 2,
            HIDDEN,
            2,
            activation=jax.nn.swish,
            key=keys[0],
        )
        cells = []
        for organ, k in zip(ORGANS, keys[2:]):
            size = sum(st.size for st in organ.states)
            cells.append(
                _zero_last_layer(
                    eqx.nn.MLP(_cell_input_size(organ), size + N_LATENT, HIDDEN, 2, activation=jax.nn.swish, key=k)
                )
            )
        decoder = eqx.nn.MLP(
            LAYOUT.size + EMBED_SIZE, len(OBS_BASE), HIDDEN, 2, activation=jax.nn.swish, key=keys[1]
        )
        return BodyModel(
            log_sys=jnp.zeros(n_sys),
            log_path=jnp.zeros((KNOWLEDGE.n_pathogens, n_path)),
            log_drug=jnp.zeros((MAX_DRUGS, n_drug)),
            log_iiv=jnp.zeros(3),
            log_obs_sigma=jnp.log(jnp.asarray(_OBS_SIGMA)),
            log_tau=jnp.log(jnp.tile(jnp.asarray(LATENT_TAUS_H), (len(ORGANS), 1))),
            encoder=_zero_last_layer(encoder),
            cells=tuple(cells),
            decoder=_zero_last_layer(decoder),
        )

    # ---- From a person's description to the numbers the equations use ----

    def context(self, cov, eta_sys, eta_path, eta_drug, drug_base) -> Context:
        norm_cov = (cov - COV_CENTER) / COV_SCALE
        enc = self.encoder(norm_cov)
        embed = enc[:EMBED_SIZE]
        n_sys = len(KNOWLEDGE.sys_names)
        delta_sys = 0.5 * jnp.tanh(enc[EMBED_SIZE : EMBED_SIZE + n_sys])
        delta_cl, delta_v = 0.5 * jnp.tanh(enc[EMBED_SIZE + n_sys :])

        iiv_sys, iiv_path, iiv_drug = jnp.exp(self.log_iiv)
        sys_vec = KNOWLEDGE.sys_values * jnp.exp(
            jnp.clip(self.log_sys, -2.3, 2.3) + delta_sys + iiv_sys * KNOWLEDGE.sys_iiv * eta_sys
        )
        path_mat = KNOWLEDGE.path_values * jnp.exp(
            jnp.clip(self.log_path, -2.3, 2.3) + iiv_path * KNOWLEDGE.path_iiv * eta_path
        )
        trained = jnp.asarray(_REGISTRY_MASK)
        drug_mult = jnp.clip(self.log_drug, -2.3, 2.3) * trained[:, None]
        drug_mat = drug_base * jnp.exp(drug_mult + iiv_drug * KNOWLEDGE.drug_iiv * eta_drug)
        cl_i, v_i = _DRUG_FIELD_INDEX["cl_ref_l_h"], _DRUG_FIELD_INDEX["v_l_per_kg"]
        drug_mat = drug_mat.at[:, cl_i].multiply(jnp.exp(delta_cl * trained))
        drug_mat = drug_mat.at[:, v_i].multiply(jnp.exp(delta_v * trained))

        patient = {name: cov[i] for i, name in enumerate(COVARIATES)}
        sex_factor = 1.0 - 0.15 * cov[COV_INDEX["sex_female"]]
        crcl = (140.0 - cov[COV_INDEX["age_years"]]) * cov[COV_INDEX["weight_kg"]] / (
            72.0 * cov[COV_INDEX["creatinine_mg_dl"]]
        )
        patient["crcl"] = jnp.clip(crcl * sex_factor, 5.0, 200.0)
        ageing = 1.0 - 0.006 * jnp.maximum(cov[COV_INDEX["age_years"]] - 40.0, 0.0)
        patient["immune_scale"] = jnp.clip(cov[COV_INDEX["immune_function"]] * ageing, 0.05, 2.0)

        return Context(
            sys={n: sys_vec[i] for i, n in enumerate(KNOWLEDGE.sys_names)},
            path={n: path_mat[:, i] for i, n in enumerate(KNOWLEDGE.path_fields)},
            drug={n: drug_mat[:, i] for i, n in enumerate(DRUG_FIELDS)},
            patient=patient,
            embed=embed,
            slot_trained=trained,
        )

    def initial_state(self, c: Context) -> jax.Array:
        y = jnp.zeros(LAYOUT.size)
        for name, off, size, st, _ in LAYOUT.entries:
            start = st.init if st.init_from is None else c.patient[st.init_from]
            y = y.at[off : off + size].set(start)
        return y

    # ---- The differential equation ----

    def vector_field(self, t, y, c: Context) -> jax.Array:
        s = LAYOUT.unpack(y)
        e = _publish_signals(s, c)
        d: dict = {}
        for organ in ORGANS:
            d.update(organ.derivative(s, c, e))

        features = {
            name: jnp.atleast_1d(_state_features(s[name], st)) for name, _, _, st, _ in LAYOUT.entries
        }
        latents = []
        for i, organ in enumerate(ORGANS):
            own = [features[st.name] for st in organ.states]
            nbr = [features[st.name] for n in organ.neighbors for st in next(o for o in ORGANS if o.name == n).states]
            z = y[LAYOUT.latent_offset[organ.name] : LAYOUT.latent_offset[organ.name] + N_LATENT]
            out = self.cells[i](jnp.concatenate(own + nbr + [z, c.embed]))
            cursor = 0
            for st in organ.states:
                raw = out[cursor : cursor + st.size] if st.size > 1 else out[cursor]
                cursor += st.size
                correction = _gate(raw, s[st.name], st)
                if st.per == PER_DRUG:
                    correction = correction * c.slot_trained  # No learning applies to hypothetical drugs.
                d[st.name] = d[st.name] + correction
            tau = jnp.exp(jnp.clip(self.log_tau[i], jnp.log(1.0), jnp.log(8760.0)))
            latents.append((jnp.tanh(out[cursor:]) - z) / tau)
        return LAYOUT.pack_derivative(d, latents)

    # ---- What can be measured ----

    def observe(self, y, c: Context) -> jax.Array:
        s = LAYOUT.unpack(y)
        e = _publish_signals(s, c)
        parts = []
        for organ in ORGANS:
            if organ.measurements:
                # Read by name, so the order an organ returns its values in cannot mislabel them.
                values = organ.observables(s, c, e)
                parts.extend(jnp.atleast_1d(values[m.name]) for m in organ.measurements)
        base = jnp.concatenate(parts)
        features = jnp.concatenate(
            [jnp.atleast_1d(_state_features(s[n], st)) for n, _, _, st, _ in LAYOUT.entries]
            + [y[LAYOUT.physio_size :], c.embed]
        )
        base = base + jnp.asarray(_RESIDUAL_SCALE) * jnp.tanh(self.decoder(features))
        shares = [sigmoid((base[OBS_INDEX[x.measurement]] - x.threshold) / x.softness) for x in SHARES]
        return jnp.concatenate([base, jnp.stack(shares)]) if shares else base

    # ---- Changes that happen at an instant ----

    def apply_events(self, y, c: Context, oral_mg, iv_mg, exposure_log10):
        """Applies doses and new pathogen exposures. Negative exposure means none."""
        gut = LAYOUT.get(y, "gut") + oral_mg * c.drug["bioavailability"]
        volume = c.drug["v_l_per_kg"] * c.patient["weight_kg"]
        conc = LAYOUT.get(y, "conc") + iv_mg / volume
        surface = LAYOUT.get(y, "log_surface")
        merged = jnp.logaddexp(surface * 2.302585, exposure_log10 * 2.302585) / 2.302585
        surface = jnp.where(exposure_log10 > -9.0, merged, surface)
        y = LAYOUT.set(y, "gut", gut)
        y = LAYOUT.set(y, "conc", conc)
        return LAYOUT.set(y, "log_surface", surface)


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = 1e-4
    atol: float = 1e-5
    dtmax: float = 1.0
    max_steps: int = 4096
    stiff: bool = False


SERVE_SOLVER = SolverConfig()
# The largest step must stay inside the solver's stability limit for the fastest states.
# At 2 h, gradients reach 100 million. At 1 h they are about 5, and they match 0.5 h.
TRAIN_SOLVER = SolverConfig(rtol=1e-3, atol=1e-3, dtmax=1.0, max_steps=2048)


# Gradients through the solver stay meaningful only with these protections, along with
# the largest step in TRAIN_SOLVER.
#
# A fast state near zero, such as a drug nobody has taken, shows the adaptive solver
# no error, so the solver may take steps too long for that state. Going forward this
# is harmless. Going backward, that state's gradient grows on every step. It can reach
# hundreds of millions and drown every real signal, or overflow, and then infinity
# times zero turns every gradient into NaN. Forward results are never affected.
GRADIENT_LIMIT = 1e6
UNRESOLVED_TOLERANCES = 10.0
_IS_AMOUNT = np.zeros(LAYOUT.size, dtype=bool)
for _name, _offset, _size, _state, _ in LAYOUT.entries:
    _IS_AMOUNT[_offset : _offset + _size] = _state.kind == AMOUNT


@jax.custom_vjp
def _limit_gradient(y):
    """Leaves values alone and keeps gradients through them finite, as a safety net."""
    return y


def _limit_gradient_forward(y):
    return y, None


def _limit_gradient_backward(_, gradient):
    return (jnp.clip(jnp.nan_to_num(gradient, nan=0.0), -GRADIENT_LIMIT, GRADIENT_LIMIT),)


_limit_gradient.defvjp(_limit_gradient_forward, _limit_gradient_backward)


def _stable_gradients(y, atol):
    """Blocks gradients through amounts too small for the solver to resolve.

    Below its error tolerance the solver is not tracking an amount, so that value is
    numerical noise and no real signal passes through it.
    """
    unresolved = jnp.logical_and(jnp.asarray(_IS_AMOUNT), jnp.abs(y) < UNRESOLVED_TOLERANCES * atol)
    return _limit_gradient(jnp.where(unresolved, jax.lax.stop_gradient(y), y))


def advance(model: BodyModel, y, c: Context, dt, cfg: SolverConfig = SERVE_SOLVER):
    """Integrates one person forward by dt hours."""
    dt_safe = jnp.maximum(dt, 1e-3)
    if cfg.stiff:
        # An implicit solver needs forward derivatives, and it does not have this problem.
        term = diffrax.ODETerm(lambda t, state, args: model.vector_field(t, state, args))
    else:
        term = diffrax.ODETerm(lambda t, state, args: model.vector_field(t, _stable_gradients(state, cfg.atol), args))
    solver = diffrax.Kvaerno5() if cfg.stiff else diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=dt_safe,
        dt0=jnp.minimum(dt_safe, 0.1),
        y0=y,
        args=c,
        stepsize_controller=diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol, dtmax=cfg.dtmax),
        max_steps=cfg.max_steps,
        throw=False,
    )
    out = sol.ys[0]
    ok = diffrax.is_successful(sol.result)
    out = jnp.where(jnp.logical_and(ok, jnp.all(jnp.isfinite(out))), out, jnp.nan * jnp.ones_like(out))
    return jnp.where(dt <= 1e-6, y, out)


def trainable_groups(model: BodyModel) -> dict:
    """Parameter counts by group, reported so contributors can see what can learn."""
    def count(tree):
        return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))))

    return {
        "physiology_multipliers": count((model.log_sys, model.log_path, model.log_drug, model.log_iiv)),
        "patient_encoder": count(model.encoder),
        "organ_cells": count(model.cells),
        "memory_time_constants": count(model.log_tau),
        "observation_network": count(model.decoder),
        "measurement_noise": count(model.log_obs_sigma),
    }
