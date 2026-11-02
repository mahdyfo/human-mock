"""Cytokines and the blood markers they drive."""

import jax.numpy as jnp

from .base import LEVEL, Measurement, Organ, StateDef


def derivative(s, c, e):
    k = c.sys
    p = c.patient
    drive = s["innate"] * e["pamp"] + k["tissue_to_cytokine"] * s["tissue"]
    return {
        "cytokine": k["k_cytokine"] * (drive - s["cytokine"]),
        "crp": k["k_crp_prod"] * s["cytokine"] - k["k_crp_off"] * (s["crp"] - p["crp_mg_l"]),
        "wbc": k["k_wbc_prod"] * s["cytokine"] - k["k_wbc_off"] * (s["wbc"] - p["wbc_k_ul"]),
    }


def observables(s, c, e):
    return {"crp_mg_l": jnp.maximum(s["crp"], 0.0), "wbc_k_ul": jnp.maximum(s["wbc"], 0.0)}


ORGAN = Organ(
    name="inflammation",
    description="Cytokine drive, C-reactive protein, and white cell count.",
    states=(
        StateDef("cytokine", LEVEL, rate_scale=0.1, scale=1.0, label="Cytokines"),
        StateDef("crp", LEVEL, rate_scale=2.0, init_from="crp_mg_l", center=10.0, scale=30.0, internal=False),
        StateDef("wbc", LEVEL, rate_scale=0.3, init_from="wbc_k_ul", center=7.0, scale=4.0, internal=False),
    ),
    neighbors=("immunity", "throat", "thermoregulation"),
    derivative=derivative,
    observables=observables,
    measurements=(
        Measurement("crp_mg_l", "mg per litre", noise=8.0, neural_range=15.0, label="C-reactive protein", body_part="blood", healthy=10.0, severe=150.0),
        Measurement("wbc_k_ul", "thousand cells per microlitre", noise=1.5, neural_range=2.0, label="White blood cells", body_part="blood", healthy=11.0, severe=25.0),
    ),
)
