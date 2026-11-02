# How the simulator is built

## The idea

A neural network that learns a whole body from data would need far more data than
exists, and it would produce confident nonsense outside what it had seen. So this
body combines equations people already trust with neural networks that learn what
those equations miss.

## One virtual person

A person is a single vector that changes over time according to one differential
equation. It has two parts.

**Physiological states** are named things: drug levels, bacterial load, immune
activation, cytokines, tissue damage, temperature.

**Memory neurons** are unnamed. Each organ owns four leaky integrators, with time
constants starting at 6 hours, 1 day, 1 week, and 2 months. They can hold on to
things the equations have no variable for. Training can adjust their time constants.

The rate of change of each physiological state is:

```
known biology from the organ module + bounded neural correction
```

## The networks

| Network | Input | Output |
|---|---|---|
| Patient encoder | Age, weight, labs, and other parameters | A personal embedding, and personal adjustments to constants and drug clearance |
| Organ cell, one per organ | That organ's states and memory neurons, connected organs' states, and the embedding | A correction per state, and a drive for each memory neuron |
| Observation network | The whole state and the embedding | A correction to each measurement |

Organs only see the organs they are physiologically connected to, which limits how
much the networks can invent.

## Why learning cannot break the body

- **Silent start.** The last layer of every network starts at zero, so an untrained
  body is exactly the equations.
- **Bounded corrections.** Each state declares the largest correction per hour.
- **Physical gates.** A drug correction is proportional to the drug present, so no
  drug appears from nothing. A pathogen at extinction stays extinct.
- **Prior pull.** Training penalises moving physiological constants far from their
  starting values.

## Three kinds of memory

| Memory | Where it lives | What it does |
|---|---|---|
| Session memory | SQLite snapshots on disk | Resume, reset, fork, and review, even after a restart |
| Biological memory | The immune memory state | A second infection is milder, because the body remembers |
| Neural memory | The memory neurons | Trained slow effects with no named variable |

All three are saved in each snapshot. A reset restores all three, so a body really
forgets what happened after the step it returns to.

## Uncertainty

A session holds a cloud of virtual people rather than one person. Parameters the
caller left out are sampled from similar people in NHANES, and each person gets their
own physiological variation. Results are reported as the 10th, 50th, and 90th
percentiles across that cloud. An ensemble of trained models spreads people across
models, so disagreement between models widens the results too.

Every response also says whether the request stays within what the model was trained
on. An AI agent searching for treatments should treat results outside that support
as unreliable, because an optimiser is naturally drawn to wherever a model is wrong.

## Keeping gradients meaningful

Training sends gradients backward through the solver, and this can fail without any
visible sign in the results. The adaptive solver chooses its steps by the errors it
sees. A fast state sitting at zero, such as a drug nobody has taken, shows no error, so
the solver may step too far for it. Going forward nothing goes wrong. Going backward,
that state's gradient grows on every step until it drowns every real signal or overflows.

Three things prevent this, and none of them changes a forward result:

- **The largest training step is 1 hour,** inside the stability limit of the fastest
  states. At 2 hours, gradients reach hundreds of millions. At 1 hour they are about 5,
  and they match the values at 30 minutes.
- **Amounts too small for the solver to resolve pass no gradient,** because below its
  tolerance the value is numerical noise.
- **A cap keeps every gradient through the solver finite,** as a safety net.

Training counts any step it has to skip, and a test in `tests/test_contract.py` fails
for any organ that makes gradients explode.

## Time

Doses and exposures are instant jumps applied at segment boundaries. Between them,
the differential equation is solved with an adaptive method. This makes advancing
8 hours once match advancing 1 hour eight times, and a test enforces that.

## Files

| Path | Purpose |
|---|---|
| `src/humanmock/organs/` | The biology, one file per organ |
| `src/humanmock/knowledge_data/physiology.yaml` | Patient parameters, constants, illnesses, and drugs, with sources |
| `examples/add_an_organ/` | A complete worked example of adding an organ |
| `src/humanmock/model.py` | Assembles organs and networks into one equation |
| `src/humanmock/population.py` | Fills in missing parameters |
| `src/humanmock/engine.py` | Runs virtual people through time |
| `src/humanmock/sessions.py` | Session memory |
| `src/humanmock/api.py` | The HTTP API |
| `src/humanmock/ui/` | The web page, in plain HTML, CSS, and JavaScript |
| `src/humanmock/data/` | Downloads and converters |
| `src/humanmock/training/train.py` | The training loop |
| `scripts/behaviour_report.py` | Standard scenarios against known targets |
| `tests/test_contract.py` | Rules every organ and constant must follow, applied automatically |
| `tests/test_contributor_guide.py` | Follows the contributor guide, so it cannot go out of date |

## How contributions plug in

Everything outside `organs/` and `physiology.yaml` reads their declarations, so biology
can grow without changing the rest of the code.

| Declared in | Reaches |
|---|---|
| `StateDef` in an organ | The solver, a neural cell, the physics gates, `include_internal`, and the page |
| `Measurement` in an organ | The observation network, data files, training, API responses, charts, and the body diagram |
| `Share` in an organ | Trial outcomes in data files, `share_of_people` in responses, and chart thresholds |
| `patient:` in the YAML | The API, data files, missing value sampling, and the patient encoder |
| `drugs:` in the YAML | The API, data files, and the number of drug slots |
| `pathogens:` in the YAML | The API, data files, and every per-pathogen state |

Each of these changes the shape of the body, so a trained model records the shape it
was trained on and refuses to load into a different one.
