"""Body temperature and the vital signs that follow it."""

import jax.numpy as jnp

from .base import FREE, Organ, StateDef


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
    states=(StateDef("temperature", FREE, rate_scale=0.5, init=36.7, center=37.0, scale=1.0),),
    neighbors=("inflammation", "pharmacology"),
    derivative=derivative,
    observables=observables,
    observable_names=("temperature_c", "heart_rate_bpm", "respiratory_rate_bpm"),
)
