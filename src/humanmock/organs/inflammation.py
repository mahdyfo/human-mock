"""Cytokines and the blood markers they drive."""

import jax.numpy as jnp

from .base import LEVEL, Organ, StateDef


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
        StateDef("cytokine", LEVEL, rate_scale=0.1, scale=1.0),
        StateDef("crp", LEVEL, rate_scale=2.0, init=1.0, center=10.0, scale=30.0),
        StateDef("wbc", LEVEL, rate_scale=0.3, init=7.0, center=7.0, scale=4.0),
    ),
    neighbors=("immunity", "throat", "thermoregulation"),
    derivative=derivative,
    observables=observables,
    observable_names=("crp_mg_l", "wbc_k_ul"),
)
