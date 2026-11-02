"""Local throat tissue: damage, healing, pain, and cough."""

import jax.numpy as jnp

from .base import FRACTION, LEVEL, Organ, StateDef


def derivative(s, c, e):
    path = c.path
    insult = jnp.sum(path["damage_rate"] * e["surface_activity"])
    cough_drive = jnp.sum(path["cough_potency"] * e["surface_activity"])
    return {
        "tissue": insult * (1.0 - s["tissue"]) - c.sys["k_heal"] * s["tissue"],
        "cough": c.sys["k_cough"] * (cough_drive - s["cough"]),
    }


def observables(s, c, e):
    k = c.sys
    exposure = k["pain_tissue"] * jnp.maximum(s["tissue"], 0.0) + k["pain_cytokine"] * jnp.maximum(s["cytokine"], 0.0)
    pain = 10.0 * (1.0 - jnp.exp(-exposure)) * (1.0 - e["analgesic"])
    cough = 10.0 * (1.0 - jnp.exp(-2.0 * jnp.maximum(s["cough"], 0.0)))
    return {
        "throat_pain_0_10": pain,
        "cough_0_10": cough,
    }


ORGAN = Organ(
    name="throat",
    description="Tissue damage with slow healing, which is why pain outlasts the pathogen.",
    states=(
        StateDef("tissue", FRACTION, rate_scale=0.03),
        StateDef("cough", LEVEL, rate_scale=0.05),
    ),
    neighbors=("pathogen", "inflammation"),
    derivative=derivative,
    observables=observables,
    observable_names=("throat_pain_0_10", "cough_0_10"),
)
