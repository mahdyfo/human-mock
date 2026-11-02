"""Whole-body consequences: fatigue, gut flora disturbance, and organ stress."""

import jax.numpy as jnp

from .base import FRACTION, LEVEL, Measurement, Organ, StateDef


def derivative(s, c, e):
    k = c.sys
    return {
        "fatigue": k["k_fatigue"] * (s["cytokine"] - s["fatigue"]),
        "dysbiosis": e["gut_insult"] * (1.0 - s["dysbiosis"]) - k["k_gut_recover"] * s["dysbiosis"],
        "toxicity": e["tox_insult"] * (1.0 - s["toxicity"]) - k["k_tox_recover"] * s["toxicity"],
    }


def observables(s, c, e):
    g = jnp.maximum(s["dysbiosis"], 0.0)
    return {
        "fatigue_0_10": 10.0 * (1.0 - jnp.exp(-c.sys["fatigue_gain"] * jnp.maximum(s["fatigue"], 0.0))),
        "diarrhea_0_10": 10.0 * g**2 / (g**2 + 0.25**2),
        "toxicity_index_0_1": jnp.clip(s["toxicity"], 0.0, 1.0),
    }


ORGAN = Organ(
    name="systemic",
    description="Slow whole-body states that make side effects and overdoses visible.",
    states=(
        StateDef("fatigue", LEVEL, rate_scale=0.03),
        StateDef("dysbiosis", FRACTION, rate_scale=0.005),
        StateDef("toxicity", FRACTION, rate_scale=0.01),
    ),
    neighbors=("inflammation", "pharmacology", "thermoregulation"),
    derivative=derivative,
    observables=observables,
    measurements=(
        Measurement("fatigue_0_10", "score 0 to 10", noise=1.0, neural_range=2.0),
        Measurement("diarrhea_0_10", "score 0 to 10", noise=1.0, neural_range=2.0),
        Measurement("toxicity_index_0_1", "index 0 to 1", noise=0.1),
    ),
)
