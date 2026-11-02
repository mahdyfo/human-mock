# humanmock

[![tests](https://github.com/mahdyfo/human-mock/actions/workflows/tests.yml/badge.svg)](https://github.com/mahdyfo/human-mock/actions/workflows/tests.yml)

An open, trainable simulator of human body behaviour, served through an API with
session memory.

You describe a person, give their body doses or exposures, and step time forward.
The body responds the way the current model says it should, and it remembers
everything between calls until you reset it.

> **This is a research simulator, not a medical tool.** Do not use it to diagnose or
> treat anyone. Most of its constants are still assumptions, and it covers only one
> kind of illness so far. Each response says whether the request is within what the
> model was trained on.

## What exists today

- **Seven organ modules:** drug handling, pathogens, immunity, inflammation, throat
  tissue, temperature and vital signs, and whole-body effects.
- **Two illnesses:** strep throat and a generic viral sore throat.
- **Four drugs:** amoxicillin, penicillin V, ibuprofen, and paracetamol, plus any
  hypothetical drug you describe by mechanism.
- **Neural cells on every organ,** with memory neurons, a patient encoder, and an
  observation network. They start silent and learn from data.
- **Defaults fitted to real people.** Missing parameters are sampled from similar
  people in the NHANES health survey.
- **Session memory.** Continue, reset, rewind, fork, and review. Sessions survive a
  server restart.
- **A web page.** A body diagram and charts that update as time moves forward.

## How far it has got

- **The neural parts can learn.** Practice data comes from a body with hidden laws, such
  as older people clearing paracetamol more slowly. No equation says this, and the
  networks find it on people they never trained on. A test requires this on every
  training change.
- **It does not yet simulate a real body well.** It has 15 real training subjects and
  one family of illness. Its predictions moved toward published trial results, but
  got worse on one study it never saw. [docs/TRAINING.md](docs/TRAINING.md#what-to-expect)
  has the numbers.
- **Nobody can die in it.** It has no severe poisoning or death, so even a dangerous
  overdose ends in recovery. Never read a result as proof that something is safe.

Getting further needs more data and more organs, which is why contributions matter.

## Help build it

This is a starting point, and it improves when people add to it. Most contributions
need no machine learning knowledge. One cited number from a paper is a real help.

- **Find a task:** `humanmock evidence` lists every constant that still needs a source.
- **Add data:** add three CSV files to a folder, then run `humanmock data check` on it.
- **Add a drug or illness:** edit one YAML file.
- **Add an organ:** write one Python file and add it to a list. It gets its own neural
  cell, memory neurons, API output, and automatic checks.

[CONTRIBUTING.md](CONTRIBUTING.md) shows each of these step by step.

## Run it

You need [Python 3.11 or newer](https://www.python.org/downloads/). Download this project,
then:

- **On Windows,** double-click `start.bat`.
- **On macOS or Linux,** run `./start.sh`.

The first time, it sets itself up, which takes a few minutes. After that it starts
straight away and opens the page in your browser. Everything stays on your computer.
Close the window, or press Ctrl+C in it, to stop.

Press "Try an example", give a medicine, and move time forward. "More options" in each section holds everything else, such as other
measurements, seeds, and inventing a drug. Click anywhere on the chart to look at that
moment, go back to it, or try something else from it and compare the two paths.

Every illness, drug, measurement, and patient parameter on the page comes from the
body's own declarations, so anything added to the body appears there by itself.

To use a command line instead, or to work on the code:

```powershell
git clone https://github.com/mahdyfo/human-mock
cd human-mock
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -m "not slow"
humanmock start
```

On macOS or Linux, activate with `source .venv/bin/activate`.

`humanmock start` uses your newest trained model, finds a free port, and opens your
browser. Until you train a model, it uses the untrained body, and the page says so.

## Use the API

Interactive documentation is at `/docs` on the same address. Use `humanmock serve` to run
only the server, on a port you choose.

Create a session. Every patient field is optional.

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions -H "content-type: application/json" -d '{
  "patient": {"age_years": 30, "sex": "male", "weight_kg": 75},
  "conditions": [{"name": "strep_a", "days_since_symptom_onset": 1}],
  "people": 64,
  "seed": 42
}'
```

Give a dose and move forward one hour. Use the `session_id` from the response.

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions/SESSION_ID/step -H "content-type: application/json" -d '{
  "duration": "1h",
  "events": [{"drug": "amoxicillin", "dose_mg": 500}]
}'
```

After one hour the drug is in the blood and bacteria are falling, but symptoms have
hardly moved. Keep calling with `"duration": "8h"` and the same dose to follow a
course of treatment, or send a whole schedule in one call:

```json
{"duration": "10d", "events": [{"drug": "amoxicillin", "dose_mg": 500, "repeat": 30, "every_hours": 8}]}
```

Each response reports the spread across the virtual people:

```json
{
  "session_id": "s_a1d50c3d551c0aef",
  "step": 10,
  "time_hours": 72.0,
  "virtual_people": 64,
  "failed_virtual_people": 0,
  "observations": {
    "temperature_c": {"p10": 36.7, "p50": 36.9, "p90": 37.2},
    "throat_pain_0_10": {"p10": 1.4, "p50": 2.3, "p90": 3.4}
  },
  "drug_levels_mg_l": {"amoxicillin": {"p10": 0.1, "p50": 0.17, "p90": 0.3}},
  "share_of_people": {"sore_throat": 0.58, "fever": 0.0, "culture_positive": 0.0},
  "warnings": [],
  "support": {"within_training_support": false, "reasons": ["The model has not been trained on data yet."]},
  "model": {"id": "prior-v0", "status": "prior_only", "note": "..."}
}
```

This example is abbreviated and its numbers are illustrative. Add `"include_internal": true` to a step to
see hidden states such as bacterial load and immune activation.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/sessions` | Create a session |
| `POST /v1/sessions/{id}/step` | Apply events and move time forward |
| `GET /v1/sessions/{id}` | Read the current state |
| `POST /v1/sessions/{id}/reset` | Forget everything after `to_step`, which defaults to the start |
| `POST /v1/sessions/{id}/fork` | Copy a session to try something else on identical bodies |
| `GET /v1/sessions/{id}/history` | Read every step so far |
| `DELETE /v1/sessions/{id}` | Delete a session |
| `GET /v1/sessions` | List sessions |
| `GET /v1/catalog` | List drugs, conditions, and measurements |
| `GET /v1/model` | Describe the loaded model and its training data |

### Reset options

```json
{"to_step": 0}
{"to_step": 3}
{"new_people": true, "seed": 7}
```

The first returns to the start with the same virtual people. The second rewinds to
step 3. The third draws new virtual people from the same description.

### A hypothetical drug

Describe a drug by mechanism when you create the session.

```json
{
  "conditions": [{"name": "strep_a"}],
  "custom_drugs": [{
    "name": "candidate_x",
    "ka": 1.0, "v_l_per_kg": 0.5, "cl_ref_l_h": 10,
    "kill_emax": 2.0, "kill_ec50": 0.05, "reservoir_penetration": 0.5
  }]
}
```

Results for a hypothetical drug only restate the parameters you supplied. They are
hypotheses, never evidence that such a drug would work.

## Train it

The full guide is in [docs/TRAINING.md](docs/TRAINING.md).

```powershell
humanmock data all
humanmock train --out checkpoints/run1
humanmock evaluate checkpoints/run1
humanmock serve --checkpoint checkpoints/run1
```

## Learn more

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the body and networks work.
- [docs/TRAINING.md](docs/TRAINING.md) covers data, training, and honest evaluation.
- [CONTRIBUTING.md](CONTRIBUTING.md) shows how to add organs, drugs, and evidence.

## Evidence used so far

- [NHANES 2017-2018](https://wwwn.cdc.gov/nchs/nhanes/) for population values.
- [PK-DB](https://pk-db.com) for paracetamol concentration curves.
- [DailyMed amoxicillin label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=e82de7ec-0ca3-4afb-aa09-5ec5e7895b9d) for amoxicillin kinetics.
- [Cochrane review of antibiotics for sore throat, 2021](https://www.cochrane.org/evidence/CD000023_antibiotics-adults-and-children-sore-throats) for recovery rates.
- [Review of time to negative throat culture](https://pmc.ncbi.nlm.nih.gov/articles/PMC10103550/) for bacterial clearance.
- [EUCAST S. pyogenes calibration data](https://www.eucast.org/fileadmin/eucast/pdf/disk_criteria/validation_2026/S._pyogenes_v_10.0_January_2026.pdf) for penicillin susceptibility.

Every constant is listed with its source in
`src/humanmock/knowledge_data/physiology.yaml`.

## Licence

The code is released under the [MIT licence](LICENSE). Downloaded datasets keep their
own licences, which is why they are not stored in this repository.
