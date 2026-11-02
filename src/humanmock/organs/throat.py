"""Local throat tissue: damage, healing, pain, and cough."""

import jax.numpy as jnp

from .base import FRACTION, LEVEL, Measurement, Organ, Share, StateDef


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
        StateDef("tissue", FRACTION, rate_scale=0.03, label="Throat tissue damage"),
        StateDef("cough", LEVEL, rate_scale=0.05, internal=False),
    ),
    neighbors=("pathogen", "inflammation"),
    derivative=derivative,
    observables=observables,
    measurements=(
        Measurement("throat_pain_0_10", "score 0 to 10", noise=1.0, neural_range=2.0, label="Throat pain", body_part="throat", severe=8.0),
        Measurement("cough_0_10", "score 0 to 10", noise=1.0, neural_range=2.0, label="Cough", body_part="lungs", severe=8.0),
    ),
    shares=(Share("sore_throat_fraction", "throat_pain_0_10", 2.0, 0.5, noise=0.1, label="Sore throat"),),
)
