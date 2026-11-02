"""The contract every organ module follows.

An organ declares its state variables, the organs it talks to, the equations it
knows, and what can be measured. The body assembles organs into one differential
equation and gives each organ its own neural cell and memory neurons. Everything
else, including the API, the data format, and training, reads this declaration,
so a contributor adds biology without touching any other code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from humanmock.knowledge import KNOWLEDGE, MAX_DRUGS

# How a state may be corrected by neural networks, so learning cannot break physics.
AMOUNT = "amount"  # Non-negative and cannot appear from nothing, such as a drug level.
COUNT = "count"  # log10 population that cannot appear from nothing, such as bacteria.
FRACTION = "fraction"  # Stays between 0 and 1.
LEVEL = "level"  # Non-negative with a baseline, such as CRP.
FREE = "free"  # Real-valued, such as temperature.
KINDS = (AMOUNT, COUNT, FRACTION, LEVEL, FREE)

PER_DRUG = "drug"
PER_PATHOGEN = "pathogen"


@dataclass(frozen=True)
class StateDef:
    """One thing the body keeps track of over time."""

    name: str
    kind: str
    rate_scale: float
    """Largest correction per hour a neural cell may apply to this state."""
    per: str | None = None
    """PER_DRUG or PER_PATHOGEN holds one value for each. None holds a single value."""
    init: float = 0.0
    init_from: str | None = None
    """A patient parameter that sets the starting value instead of init."""
    center: float = 0.0
    scale: float = 1.0

    @property
    def size(self) -> int:
        return {None: 1, PER_DRUG: MAX_DRUGS, PER_PATHOGEN: KNOWLEDGE.n_pathogens}[self.per]


@dataclass(frozen=True)
class Measurement:
    """Something that can be measured. It becomes usable in data files and API responses."""

    name: str
    unit: str
    noise: float = 1.0
    """Starting measurement noise, in the same unit. Training adjusts it."""
    neural_range: float = 0.0
    """Largest correction the observation network may apply. Zero reads the equations directly."""
    public: bool = True
    """Whether API responses list it under observations."""


@dataclass(frozen=True)
class Share:
    """The share of people past a threshold, for trial outcomes such as "40% had a fever"."""

    name: str
    measurement: str
    threshold: float
    softness: float
    """How gradually the threshold applies during training. Responses use a hard cut-off."""
    noise: float = 0.1


@dataclass(frozen=True)
class Organ:
    name: str
    states: tuple[StateDef, ...]
    neighbors: tuple[str, ...]
    """The organs whose states this organ's neural cell may see. Keep these to real physiological links."""
    derivative: Callable
    """derivative(s, c, e) returns the rate of change of every state this organ owns."""
    signals: Callable | None = None
    """signals(s, c, e) publishes values into e, for organs listed below this one."""
    observables: Callable | None = None
    """observables(s, c, e) returns a value for every name in measurements."""
    measurements: tuple[Measurement, ...] = ()
    shares: tuple[Share, ...] = ()
    description: str = ""


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -30.0, 30.0)))


def hill(conc, ec50, h=1.0):
    """Saturating dose-response. Safe at zero concentration."""
    c = jnp.maximum(conc, 0.0) + 1e-9
    return c**h / (ec50**h + c**h)
