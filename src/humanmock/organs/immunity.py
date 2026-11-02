"""Innate activation, adaptive effectors, and long-lived immune memory."""

import jax.numpy as jnp

from .base import FRACTION, PER_PATHOGEN, Organ, StateDef


def signals(s, c, e):
    e["pamp"] = jnp.sum(c.path["pyrogen"] * e["surface_activity"])


def derivative(s, c, e):
    k = c.sys
    immune = c.patient["immune_scale"]
    activation = k["k_effector_on"] * immune * e["antigen"] * (1.0 + k["memory_boost"] * s["memory"])
    return {
        "innate": k["k_innate_on"] * immune * e["pamp"] * (1.0 - s["innate"]) - k["k_innate_off"] * s["innate"],
        "effector": activation * (1.0 - s["effector"]) - k["k_effector_off"] * s["effector"],
        "memory": k["k_memory_on"] * s["effector"] * (1.0 - s["memory"]) - k["k_memory_off"] * s["memory"],
    }


ORGAN = Organ(
    name="immunity",
    description="Innate response, pathogen-specific effectors, and pathogen-specific memory.",
    states=(
        StateDef("innate", FRACTION, rate_scale=0.05, label="Fast immune response"),
        StateDef("effector", FRACTION, rate_scale=0.02, per=PER_PATHOGEN, label="Targeted immune response"),
        StateDef("memory", FRACTION, rate_scale=0.002, per=PER_PATHOGEN, label="Immune memory"),
    ),
    neighbors=("pathogen", "inflammation"),
    derivative=derivative,
    signals=signals,
)
