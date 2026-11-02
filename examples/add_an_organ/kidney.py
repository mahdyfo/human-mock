"""An example organ: kidney injury from drug exposure, with slow recovery.

Copy this file into src/humanmock/organs/ to start your own. CONTRIBUTING.md walks
through it, and tests/test_contributor_guide.py checks that these steps keep working.
"""

from .base import FRACTION, Measurement, Organ, Share, StateDef


def derivative(s, c, e):
    """The rate of change of every state this organ owns."""
    return {
        "kidney_injury": e["tox_insult"] * (1.0 - s["kidney_injury"])
        - c.sys["k_kidney_recover"] * s["kidney_injury"],
    }


def observables(s, c, e):
    """A value for every measurement declared below."""
    return {"creatinine_rise_mg_dl": c.sys["creatinine_per_injury"] * s["kidney_injury"]}


ORGAN = Organ(
    name="kidney",
    description="Injury from drug exposure, with slow recovery.",
    states=(StateDef("kidney_injury", FRACTION, rate_scale=0.01),),
    neighbors=("pharmacology", "systemic"),
    derivative=derivative,
    observables=observables,
    measurements=(Measurement("creatinine_rise_mg_dl", "mg per decilitre", noise=0.2, neural_range=0.1),),
    shares=(Share("kidney_injury_fraction", "creatinine_rise_mg_dl", 0.3, 0.05),),
)
