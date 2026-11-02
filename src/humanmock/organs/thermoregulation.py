"""Body temperature and the vital signs that follow it."""

import jax.numpy as jnp

from .base import FREE, Measurement, Organ, Share, StateDef


def derivative(s, c, e):
    k = c.sys
    cyt = jnp.maximum(s["cytokine"], 0.0)
    rise = k["fever_max"] * cyt / (cyt + k["fever_half"]) * (1.0 - e["antipyretic"])
    return {"temperature": (c.patient["body_temp_c"] + rise - s["temperature"]) / k["tau_temp"]}


def observables(s, c, e):
    k = c.sys
    p = c.patient
    excess = s["temperature"] - p["body_temp_c"]
    cyt = jnp.maximum(s["cytokine"], 0.0)
    return {
        "temperature_c": s["temperature"],
        "heart_rate_bpm": p["heart_rate_bpm"] + k["hr_per_c"] * excess + k["hr_per_cytokine"] * cyt,
        "respiratory_rate_bpm": p["respiratory_rate_bpm"] + k["rr_per_c"] * excess + k["rr_per_cytokine"] * cyt,
    }


ORGAN = Organ(
    name="thermoregulation",
    description="Fever follows a cytokine-driven set point, and antipyretic drugs lower that drive.",
    states=(StateDef("temperature", FREE, rate_scale=0.5, init_from="body_temp_c", center=37.0, scale=1.0, internal=False),),
    neighbors=("inflammation", "pharmacology"),
    derivative=derivative,
    observables=observables,
    measurements=(
        Measurement("temperature_c", "degrees C", noise=0.3, neural_range=0.5, label="Temperature", body_part="skin", healthy=37.5, severe=40.5),
        Measurement("heart_rate_bpm", "beats per minute", noise=6.0, neural_range=8.0, label="Heart rate", body_part="heart", healthy=90.0, severe=150.0),
        Measurement("respiratory_rate_bpm", "breaths per minute", noise=2.0, neural_range=3.0, label="Breathing rate", body_part="lungs", healthy=20.0, severe=35.0),
    ),
    shares=(Share("fever_fraction", "temperature_c", 38.0, 0.15, noise=0.1, label="Fever"),),
)
