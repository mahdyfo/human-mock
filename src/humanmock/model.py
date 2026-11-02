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
from humanmock.organs.base import AMOUNT, COUNT, FRACTION, LEVEL, sigmoid

COVARIATES: tuple[str, ...] = (
    "age_years",
    "sex_female",
    "weight_kg",
    "height_cm",
    "creatinine_mg_dl",
    "alt_u_l",
    "heart_rate_bpm",
    "respiratory_rate_bpm",
    "wbc_k_ul",
    "crp_mg_l",
    "body_temp_c",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "immune_function",
    "hepatic_function",
)
COV_INDEX = {n: i for i, n in enumerate(COVARIATES)}
# Rough centre and spread used only to put inputs on a similar scale for the networks.
COV_CENTER = np.array([45, 0.5, 75, 168, 0.9, 22, 72, 15, 7, 2, 36.7, 120, 72, 1, 1], dtype=np.float32)
COV_SCALE = np.array([20, 0.5, 20, 12, 0.3, 15, 12, 3, 2.5, 5, 0.4, 18, 12, 0.3, 0.3], dtype=np.float32)

N_LATENT = 4
LATENT_TAUS_H = np.array([6.0, 24.0, 168.0, 1440.0], dtype=np.float32)
EMBED_SIZE = 8
HIDDEN = 32

# Observables computed after neural correction, as (source observable, threshold, softness).
FRACTIONS: dict[str, tuple[str, float, float]] = {
    "sore_throat_fraction": ("throat_pain_0_10", 2.0, 0.5),
    "fever_fraction": ("temperature_c", 38.0, 0.15),
    "culture_positive_fraction": ("throat_culture_log10_cfu", 2.0, 0.3),
}

# Largest correction the observation network may apply. Zero means measured directly from state.
OBS_RESIDUAL_SCALE: dict[str, float] = {
    "temperature_c": 0.5,
    "heart_rate_bpm": 8.0,
    "respiratory_rate_bpm": 3.0,
    "throat_pain_0_10": 2.0,
    "cough_0_10": 2.0,
    "fatigue_0_10": 2.0,
    "diarrhea_0_10": 2.0,
    "crp_mg_l": 15.0,
    "wbc_k_ul": 2.0,
}
# Starting measurement noise per observable, learned during training.
OBS_SIGMA: dict[str, float] = {
    "temperature_c": 0.3,
    "heart_rate_bpm": 6.0,
    "respiratory_rate_bpm": 2.0,
    "throat_pain_0_10": 1.0,
    "cough_0_10": 1.0,
    "fatigue_0_10": 1.0,
    "diarrhea_0_10": 1.0,
    "crp_mg_l": 8.0,
    "wbc_k_ul": 1.5,
    "throat_culture_log10_cfu": 1.0,
    "toxicity_index_0_1": 0.1,
    "sore_throat_fraction": 0.1,
    "fever_fraction": 0.1,
    "culture_positive_fraction": 0.05,
}


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
        names = [e[0] for e in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("Two organs declare a state with the same name.")
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

OBS_BASE: tuple[str, ...] = tuple(n for o in ORGANS for n in o.observable_names)
OBS_NAMES: tuple[str, ...] = OBS_BASE + tuple(FRACTIONS.keys())
OBS_INDEX = {n: i for i, n in enumerate(OBS_NAMES)}
_RESIDUAL_SCALE = np.array([OBS_RESIDUAL_SCALE.get(n, 0.0) for n in OBS_BASE], dtype=np.float32)
_OBS_SIGMA = np.array([OBS_SIGMA.get(n, 1.0) for n in OBS_NAMES], dtype=np.float32)
_FRACTION_SPEC = [(OBS_INDEX[src], thr, soft) for (src, thr, soft) in FRACTIONS.values()]

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
            y = y.at[off : off + size].set(st.init)
        y = LAYOUT.set(y, "temperature", c.patient["body_temp_c"])
        y = LAYOUT.set(y, "crp", c.patient["crp_mg_l"])
        y = LAYOUT.set(y, "wbc", c.patient["wbc_k_ul"])
        return y

    # ---- The differential equation ----

    def vector_field(self, t, y, c: Context) -> jax.Array:
        s = LAYOUT.unpack(y)
        e: dict = {}
        for organ in ORGANS:
            if organ.signals is not None:
                organ.signals(s, c, e)
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
                if organ.name == "pharmacology":
                    correction = correction * c.slot_trained  # No learning applies to hypothetical drugs.
                d[st.name] = d[st.name] + correction
            tau = jnp.exp(jnp.clip(self.log_tau[i], jnp.log(1.0), jnp.log(8760.0)))
            latents.append((jnp.tanh(out[cursor:]) - z) / tau)
        return LAYOUT.pack_derivative(d, latents)

    # ---- What can be measured ----

    def observe(self, y, c: Context) -> jax.Array:
        s = LAYOUT.unpack(y)
        e: dict = {}
        for organ in ORGANS:
            if organ.signals is not None:
                organ.signals(s, c, e)
        parts = []
        for organ in ORGANS:
            if organ.observables is not None:
                parts.extend(jnp.atleast_1d(v) for v in organ.observables(s, c, e).values())
        base = jnp.concatenate(parts)
        features = jnp.concatenate(
            [jnp.atleast_1d(_state_features(s[n], st)) for n, _, _, st, _ in LAYOUT.entries]
            + [y[LAYOUT.physio_size :], c.embed]
        )
        base = base + jnp.asarray(_RESIDUAL_SCALE) * jnp.tanh(self.decoder(features))
        fractions = [sigmoid((base[i] - thr) / soft) for i, thr, soft in _FRACTION_SPEC]
        return jnp.concatenate([base, jnp.stack(fractions)])

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
TRAIN_SOLVER = SolverConfig(rtol=1e-3, atol=1e-3, dtmax=2.0, max_steps=512)


def advance(model: BodyModel, y, c: Context, dt, cfg: SolverConfig = SERVE_SOLVER):
    """Integrates one person forward by dt hours."""
    dt_safe = jnp.maximum(dt, 1e-3)
    term = diffrax.ODETerm(lambda t, state, args: model.vector_field(t, state, args))
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
