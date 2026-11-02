"""Innate activation, adaptive effectors, and long-lived immune memory."""

import jax.numpy as jnp

from humanmock.knowledge import KNOWLEDGE

from .base import FRACTION, Organ, StateDef

P = KNOWLEDGE.n_pathogens


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
        StateDef("innate", FRACTION, rate_scale=0.05),
        StateDef("effector", FRACTION, rate_scale=0.02, size=P),
        StateDef("memory", FRACTION, rate_scale=0.002, size=P),
    ),
    neighbors=("pathogen", "inflammation"),
    derivative=derivative,
    signals=signals,
)
