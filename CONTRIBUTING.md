# Contributing

This project is an open attempt to simulate human body behaviour. Most useful
contributions need no machine learning knowledge. Reading one paper carefully can
be enough.

## Find something to do

| If you have | You can | What you change |
|---|---|---|
| A paper with a measured value | [Replace an assumption](#replace-an-assumption) | One line of YAML |
| Published trial or study results | [Add data](#add-data) | Three CSV files |
| Knowledge of a drug | [Add a drug](#add-a-drug) | YAML only |
| Knowledge of an infection | [Add an illness](#add-an-illness) | YAML only |
| A measurement a new organ needs | [Add a patient parameter](#add-a-patient-parameter) | One line of YAML |
| Knowledge of how an organ works | [Add an organ](#add-an-organ) | One Python file, two lines in a list, and YAML |
| A known fact about how bodies behave | [Add a behaviour test](#add-a-behaviour-test) | A short test |
| Machine learning experience | Improve training in `src/humanmock/training/train.py` | Python |

Run this for a list of constants that still need a cited source:

```powershell
humanmock evidence
```

## Set up

Requires Python 3.11 or newer. On macOS or Linux, activate with `source .venv/bin/activate`.

```powershell
git clone https://github.com/mahdyfo/human-mock
cd human-mock
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -m "not slow"
```

## Replace an assumption

Every constant lives in `src/humanmock/knowledge_data/physiology.yaml`. Find one
marked `source: assumption`, then put in the measured value and say where it came from.

```yaml
# Before
k_heal: {value: 0.022, iiv: 0.30, source: assumption, note: "throat tissue healing"}

# After
k_heal: {value: 0.031, iiv: 0.30, source: "Smith 2020, table 3, doi:10.xxxx/xxxxx", note: "throat tissue healing"}
```

If you calculated the value from a reported one, say how. `iiv` is how much the value
varies between people, as a log-normal standard deviation.

## Add data

1. Create a folder under `data/observations/`, such as `data/observations/smith_2020/`.
2. Add `subjects.csv`, `events.csv`, and `observations.csv`. The format and a worked
   example are in [docs/TRAINING.md](docs/TRAINING.md#the-data-format).
3. Check it. This reports every problem it can find in plain language.

```powershell
humanmock data check data/observations/smith_2020
```

Use the `notes` column to say which paper and table each subject came from. Training
reads the new folder automatically.

## Add a drug

Add an entry at the end of `drugs:` in `physiology.yaml`. Every value you leave out
takes a neutral default, so an antibiotic does not need any pain relief fields. The
field names are listed in `DRUG_FIELDS` in `src/humanmock/knowledge.py`. A misspelled
field stops with a message that suggests the right name.

```yaml
cefalexin:
  label: "Cefalexin"
  ka: 1.5
  v_l_per_kg: 0.3
  cl_ref_l_h: 20.0
  frac_renal: 0.9
  kill_emax: 1.6
  kill_ec50: 0.5
  sources:
    pk: "Where the absorption and clearance values came from"
    kill_ec50: assumption
```

The drug then works in the API and in data files, as `conc_cefalexin_mg_l`.

## Add an illness

Add an entry to `pathogens:` in `physiology.yaml`. Each infection needs every field the
others have, and the check tells you which are missing. An illness that is not an
infection needs its own organ. Add behaviour tests that describe what is known about
the illness before you train on it.

## Add a patient parameter

Add a line to `patient:` in `physiology.yaml`. It then becomes an optional API field, a
data column, and a neural network input. The `prior` says how to sample it when a
caller leaves it out.

```yaml
urine_output_l_day: {center: 1.5, scale: 0.5, min: 0.1, max: 5, note: "Daily urine output", prior: [lognormal, 0.4, 0.3], source: "Where this range came from"}
```

## Add an organ

The complete example is in [examples/add_an_organ/](examples/add_an_organ/). It takes
three steps, and a test follows them so they keep working.

**1. Copy [`kidney.py`](examples/add_an_organ/kidney.py) to `src/humanmock/organs/` and change it.**

```python
def derivative(s, c, e):
    return {
        "kidney_injury": e["tox_insult"] * (1.0 - s["kidney_injury"])
        - c.sys["k_kidney_recover"] * s["kidney_injury"],
    }
```

Your equations read three things:

| Name | What it holds | Example |
|---|---|---|
| `s` | The state of every organ, right now | `s["cytokine"]` |
| `c.sys`, `c.path`, `c.drug`, `c.patient` | Constants and this person's description | `c.patient["weight_kg"]` |
| `e` | Signals other organs published | `e["tox_insult"]` |

**2. Add it to the list in `src/humanmock/organs/__init__.py`.**

```python
from . import immunity, inflammation, kidney, pathogen, pharmacology, systemic, thermoregulation, throat

ORGANS: tuple[Organ, ...] = (
    ...
    systemic.ORGAN,
    kidney.ORGAN,
)
```

**3. Add its constants to `physiology.yaml`, with sources.** See
[`additions.yaml`](examples/add_an_organ/additions.yaml).

That is all. You do not need to touch the neural, API, or training code. The body
gives the organ:

- A neural cell and four memory neurons, which start silent.
- A place in `include_internal` for each state.
- API output and a data column for each `Measurement`.
- A place in `share_of_people` for each `Share`.
- Automatic checks in `tests/test_contract.py`.

Choose these with care:

- **`kind`** stops learning from breaking physics. A `count` that reaches extinction
  stays extinct, and an `amount` cannot appear from nothing.
- **`rate_scale`** is the largest correction per hour the neural cell may apply. Keep it
  small compared with how fast the state really changes.
- **`neighbors`** are the organs the neural cell may see. Keep these to real
  physiological links, because they limit what the network can invent.
- **`neural_range`** is the largest correction the network may make to a measurement.
  Leave it at zero for anything read directly, such as a lab value.

## Add a behaviour test

Add something that must always be true of a body to `tests/test_body.py`, with its
source in a comment. These tests protect everyone else's work, and they are often more
valuable than new features.

```python
def test_weak_kidneys_keep_a_renally_cleared_drug_longer(sim, book):
    normal, _ = sim.create_particles(ADULT, [], book, 16, 1)
    impaired, _ = sim.create_particles(dict(ADULT, age_years=78, creatinine_mg_dl=3.0), [], book, 16, 1)
    a = sim.advance(normal, book, 6.0, [AMOX])
    b = sim.advance(impaired, book, 6.0, [AMOX])
    assert median(sim, book, b, AMOX_LEVEL) > 2 * median(sim, book, a, AMOX_LEVEL)
```

## Before you open a pull request

```powershell
pytest -m "not slow"
humanmock data check
python scripts/behaviour_report.py
```

GitHub runs the tests and the data check on every pull request. If you change training,
also run `pytest -m slow`. It checks that the networks can recover hidden biology.

Adding an organ, a drug, an illness, or a patient parameter changes the shape of the
body. Models trained before that change refuse to load and need retraining. That is
expected, and pull requests never include trained models.

## Rules for evidence

- **Cite every number.** Put the source in the `source` or `notes` field.
- **Label derived numbers.** Say when a value was calculated rather than reported.
- **Respect licences.** Do not commit data whose licence forbids it. `data/raw/` and
  `data/observations/pkdb/` are ignored by git for this reason.
- **No real patient data.** Only published summaries and openly licensed study data
  belong here. Never add records about identifiable people.

## Scope

This is a research simulator. It must not be presented or used as medical advice, as a
diagnostic tool, or as a guide to treating anyone.

By contributing, you agree that your contribution is released under the
[MIT licence](LICENSE).
