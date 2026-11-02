"""The contract every organ module follows.

An organ declares its state variables, the organs it talks to, and the
equations it knows. The body assembles organs into one differential equation
and gives each organ its own neural cell and memory neurons automatically, so a
contributor adds biology without touching the neural or API code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

# How a state may be corrected by neural networks, so learning cannot break physics.
AMOUNT = "amount"  # Non-negative and cannot appear from nothing, such as a drug level.
COUNT = "count"  # log10 population that cannot appear from nothing, such as bacteria.
FRACTION = "fraction"  # Stays between 0 and 1.
LEVEL = "level"  # Non-negative with a baseline, such as CRP.
FREE = "free"  # Real-valued, such as temperature.


@dataclass(frozen=True)
class StateDef:
    name: str
    kind: str
    rate_scale: float
    """Largest correction per hour a neural cell may apply to this state."""
    size: int = 1
    init: float = 0.0
    center: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class Organ:
    name: str
    states: tuple[StateDef, ...]
    neighbors: tuple[str, ...]
    derivative: Callable
    signals: Callable | None = None
    observables: Callable | None = None
    observable_names: tuple[str, ...] = ()
    description: str = ""


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -30.0, 30.0)))


def hill(conc, ec50, h=1.0):
    """Saturating dose-response. Safe at zero concentration."""
    c = jnp.maximum(conc, 0.0) + 1e-9
    return c**h / (ec50**h + c**h)
