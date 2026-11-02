"""Drug absorption, distribution, elimination, and the effect each drug exerts."""

import jax.numpy as jnp

from humanmock.knowledge import MAX_DRUGS

from .base import AMOUNT, Organ, StateDef, hill


def signals(s, c, e):
    d = c.drug
    conc = jnp.maximum(s["conc"], 0.0)
    free = conc * d["frac_unbound"]
    kill = d["kill_emax"] * hill(free, d["kill_ec50"], d["kill_hill"])
    e["drug_kill_surface"] = jnp.sum(kill)
    e["drug_kill_reservoir"] = jnp.sum(kill * d["reservoir_penetration"])
    e["antipyretic"] = 1.0 - jnp.prod(1.0 - d["antipyretic_emax"] * hill(conc, d["antipyretic_ec50"]))
    e["analgesic"] = 1.0 - jnp.prod(1.0 - d["analgesic_emax"] * hill(conc, d["analgesic_ec50"]))
    e["gut_insult"] = jnp.sum(d["gut_rate"] * hill(conc, d["gut_ec50"]))
    e["tox_insult"] = jnp.sum(d["tox_rate"] * hill(conc, d["tox_ec50"], 3.0))


def derivative(s, c, e):
    d = c.drug
    p = c.patient
    clearance_scale = (
        d["frac_renal"] * p["crcl"] / 120.0 + (1.0 - d["frac_renal"]) * p["hepatic_function"]
    ) * (p["weight_kg"] / 70.0) ** 0.75
    clearance = d["cl_ref_l_h"] * clearance_scale * (1.0 - c.sys["tox_clearance_loss"] * s["toxicity"])
    volume = d["v_l_per_kg"] * p["weight_kg"]
    return {
        "gut": -d["ka"] * s["gut"],
        "conc": d["ka"] * s["gut"] / volume - clearance / volume * s["conc"],
    }


def observables(s, c, e):
    return {"conc": jnp.maximum(s["conc"], 0.0)}


ORGAN = Organ(
    name="pharmacology",
    description="One-compartment oral drug kinetics per drug slot, with renal and hepatic clearance.",
    states=(
        StateDef("gut", AMOUNT, rate_scale=0.5, size=MAX_DRUGS, scale=500.0),
        StateDef("conc", AMOUNT, rate_scale=0.5, size=MAX_DRUGS, scale=5.0),
    ),
    neighbors=("systemic",),
    derivative=derivative,
    signals=signals,
    observables=observables,
    observable_names=tuple(f"conc_slot{i}" for i in range(MAX_DRUGS)),
)
