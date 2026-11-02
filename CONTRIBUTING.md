# Contributing

This project is a starting point for simulating human body behaviour in the open.
The most useful contributions are, in order:

1. **Better evidence.** Replace a `source: assumption` in
   `src/humanmock/knowledge_data/physiology.yaml` with a cited value.
2. **New data.** Add a source folder under `data/observations/`.
   See docs/TRAINING.md for the format.
3. **New organs and illnesses.** See below.
4. **Hard validation tests.** Add a behaviour that must hold to `tests/test_body.py`.

## Setting up

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -m "not slow"
```

Run `pytest` without the marker before proposing a change to training.

## Adding an organ

Create a file in `src/humanmock/organs/` and add its `ORGAN` to the list in
`organs/__init__.py`. The body gives it a neural cell and memory neurons by itself.

```python
"""Kidney function, as an example."""

from .base import FRACTION, Organ, StateDef


def derivative(s, c, e):
    return {"kidney_injury": e["tox_insult"] * (1 - s["kidney_injury"]) - 0.02 * s["kidney_injury"]}


def observables(s, c, e):
    return {"creatinine_rise_mg_dl": 3.0 * s["kidney_injury"]}


ORGAN = Organ(
    name="kidney",
    description="Injury from drug exposure, with slow recovery.",
    states=(StateDef("kidney_injury", FRACTION, rate_scale=0.01),),
    neighbors=("pharmacology", "systemic"),
    derivative=derivative,
    observables=observables,
    observable_names=("creatinine_rise_mg_dl",),
)
```

- **States** declare a kind, which stops learning from breaking physics. A `count`
  that reaches extinction stays extinct, and an `amount` cannot appear from nothing.
- **Signals** let an organ publish a value for later organs in the list.
- **Parameters** belong in `physiology.yaml`, with their source.
- **Neighbors** decide which other organs the neural cell may see. Keep this to real
  physiological links.

Changing organs changes the body structure, so older trained models will refuse to
load and need retraining.

## Adding a drug

Add an entry under `drugs:` in `physiology.yaml`, with a source for each value. The
body holds eight drug slots in total, including any hypothetical drugs a session
defines. Adding a drug also needs checkpoints to be retrained.

## Adding an illness

Add an entry under `pathogens:` in `physiology.yaml`. Illnesses that are not
infections need their own organ module. Add behaviour tests that describe what is
known about the illness before you add neural training for it.

## Rules for evidence

- **Cite every number.** Put the source in the `source` or `notes` field.
- **Label derived numbers.** Say when a value was calculated rather than reported.
- **Respect licences.** Do not commit data whose licence forbids it. `data/raw/` and
  `data/observations/pkdb/` are ignored by git for this reason.
- **No real patient data.** This project works with virtual people only.

## Scope

This is a research simulator. It must not be presented or used as medical advice,
as a diagnostic tool, or as a guide to treating anyone.
