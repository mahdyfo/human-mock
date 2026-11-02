"""The body is the ordered list of organ modules below.

To add biology, write a new module that exposes ORGAN and add it to this list.
Signals run in this order, so an organ can read signals from organs above it.
"""

from . import immunity, inflammation, pathogen, pharmacology, systemic, thermoregulation, throat
from .base import Organ

ORGANS: tuple[Organ, ...] = (
    pharmacology.ORGAN,
    pathogen.ORGAN,
    immunity.ORGAN,
    inflammation.ORGAN,
    throat.ORGAN,
    thermoregulation.ORGAN,
    systemic.ORGAN,
)
