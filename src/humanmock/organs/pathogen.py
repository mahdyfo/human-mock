"""Pathogen load on the throat surface and in a protected tissue reservoir.

The reservoir explains why symptoms can resolve within days while a full course
is still needed. Drugs that penetrate cells poorly leave a population that can
re-seed the surface if treatment stops early.
"""

import jax.numpy as jnp

from .base import COUNT, PER_PATHOGEN, Measurement, Organ, Share, StateDef, sigmoid

LN10 = 2.302585
EXTINCT_LOG10 = -1.0
EXTINCTION_DRIFT = 0.5  # log10 units per hour once a population is too small to persist.


def _exchange(rate, donor, receiver, donor_est):
    ratio = jnp.exp(jnp.clip((donor - receiver) * LN10, -10.0, 5.0))
    return rate * ratio * donor_est


def signals(s, c, e):
    path = c.path
    surface, reservoir = s["log_surface"], s["log_reservoir"]
    e["surface_activity"] = sigmoid((surface - path["surface_half"]) / 0.7)
    total = jnp.logaddexp(surface * LN10, reservoir * LN10) / LN10
    e["antigen"] = sigmoid((total - path["antigen_half"]) / 0.7)


def derivative(s, c, e):
    path = c.path
    immune = c.patient["immune_scale"]
    surface, reservoir = s["log_surface"], s["log_reservoir"]

    est_s = sigmoid((surface - 1.0) / 0.15)
    est_r = sigmoid((reservoir - 1.0) / 0.15)
    alive_s = sigmoid((surface - EXTINCT_LOG10 - 0.1) / 0.05)
    alive_r = sigmoid((reservoir - EXTINCT_LOG10 - 0.1) / 0.05)

    logistic_s = 1.0 - 10.0 ** jnp.clip(surface - path["log10_capacity"], -12.0, 1.0)
    logistic_r = 1.0 - 10.0 ** jnp.clip(reservoir - path["log10_reservoir_capacity"], -12.0, 1.0)

    protection = s["effector"] + c.sys["memory_direct_kill"] * s["memory"]
    kill_innate = path["innate_kill"] * s["innate"] * immune
    kill_adaptive = path["adaptive_kill"] * protection * immune
    kill_drug = path["antibiotic_susceptibility"] * e["drug_kill_surface"]

    # Rates below are natural-log per hour, converted to log10 at the end.
    d_surface = est_s * (path["r"] * logistic_s - kill_innate - kill_adaptive - kill_drug - path["k_to_reservoir"])
    d_surface = d_surface + _exchange(path["k_from_reservoir"], reservoir, surface, est_r)

    kill_res_adaptive = path["reservoir_kill"] * protection * immune
    kill_res_drug = path["antibiotic_susceptibility"] * e["drug_kill_reservoir"]
    d_reservoir = est_r * (
        path["reservoir_r"] * logistic_r - kill_res_adaptive - kill_res_drug - path["k_from_reservoir"]
    )
    d_reservoir = d_reservoir + _exchange(path["k_to_reservoir"], surface, reservoir, est_s)

    return {
        "log_surface": d_surface / LN10 - EXTINCTION_DRIFT * (1.0 - est_s) * alive_s,
        "log_reservoir": d_reservoir / LN10 - EXTINCTION_DRIFT * (1.0 - est_r) * alive_r,
    }


def observables(s, c, e):
    # A swab reads the highest surface load among present pathogens.
    return {"throat_culture_log10_cfu": jnp.maximum(jnp.max(s["log_surface"]), 0.0)}


ORGAN = Organ(
    name="pathogen",
    description="Pathogen load by slot on the surface and in a protected reservoir.",
    states=(
        StateDef("log_surface", COUNT, rate_scale=0.3, per=PER_PATHOGEN, init=EXTINCT_LOG10, center=3.0, scale=4.0),
        StateDef("log_reservoir", COUNT, rate_scale=0.1, per=PER_PATHOGEN, init=EXTINCT_LOG10, center=1.0, scale=3.0),
    ),
    neighbors=("pharmacology", "immunity", "throat"),
    derivative=derivative,
    signals=signals,
    observables=observables,
    measurements=(Measurement("throat_culture_log10_cfu", "log10 colony forming units", noise=1.0),),
    shares=(Share("culture_positive_fraction", "throat_culture_log10_cfu", 2.0, 0.3, noise=0.05),),
)
