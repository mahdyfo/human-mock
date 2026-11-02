# Training guide

## How training works

The body is known equations plus neural corrections. An untrained model is pure
textbook physiology, because every neural output starts at zero.

Training has two stages.

1. **Calibration** adjusts physiological constants and measurement noise.
2. **Neural training** also opens the organ cells, memory neurons, patient
   encoder, and observation network, so they can learn what the equations miss.

A model improves only as much as its data allows. See "What to expect" below.

## Quick start

Run these from the project folder with the virtual environment active.

```powershell
.venv\Scripts\Activate.ps1
humanmock data all
humanmock data status
humanmock train --out checkpoints/run1
humanmock evaluate checkpoints/run1
python scripts/behaviour_report.py --checkpoint checkpoints/run1
humanmock serve --checkpoint checkpoints/run1
```

## Where data goes

```
data/
  raw/                     Downloads, exactly as received. Not committed to git.
    nhanes/                NHANES survey files
    pkdb/                  PK-DB study tables, one folder per drug
    clinicaltrials/        Trial records and a review sheet
  observations/            Everything training reads, one folder per source
    literature/            Published figures, typed in with their sources
    pkdb/                  Written by "humanmock data pkdb"
    custom/                Your own data
    synthetic/             Practice data, ignored unless you ask for it
```

Training reads every folder under `data/observations/`, except `synthetic`.
To add a source, create a new folder there with the three files described below.

## Data sources

| Source | What it trains | How to get it | Licence notes |
|---|---|---|---|
| NHANES | Defaults for missing parameters | `humanmock data nhanes` | Public US government data |
| PK-DB | Drug absorption and clearance | `humanmock data pkdb --drugs paracetamol` | Open-licence studies only by default |
| Published trials and reviews | Symptoms, cure rates, side effects | Read the paper, then type into `data/observations/custom/` | Facts may be used, cite the source |
| ClinicalTrials.gov | Trial outcomes to curate by hand | `humanmock data clinicaltrials --condition pharyngitis` | Public registry |
| Vivli or YODA | Patient-level trial records | Apply at vivli.org or yoda.yale.edu | Needs an approved proposal |
| MIMIC-IV and eICU | Hospital time series, for future modules | PhysioNet credentialed access at physionet.org | Never send to hosted AI tools without verifying the rules |

### PK-DB details

PK-DB lists a licence for each study. The open set currently includes paracetamol
but no amoxicillin or ibuprofen studies. Each of those has one closed study.

```powershell
humanmock data pkdb --drugs paracetamol
humanmock data pkdb --drugs paracetamol amoxicillin ibuprofen --include-closed
```

The second command fetches closed-licence studies for your own training. Read
`data/raw/pkdb/TERMS_OF_USE.md` first and do not redistribute that data.

By default the converter drops curves where another drug was given at the same
time. Add `--keep-comedication` to keep them.

### ClinicalTrials.gov details

Results reported on this registry come in many shapes, so a script cannot convert
them safely. The download writes a review sheet at
`data/raw/clinicaltrials/<condition>_review_sheet.csv`.

Open each promising link, decide whether the outcome matches a model variable, and
type it into `data/observations/custom/`. Few strep trials post results here, so
try wider terms such as `pharyngitis` and `tonsillitis`.

### MIMIC-IV and eICU

These need completed training and a signed agreement on PhysioNet. They describe
intensive care patients, who are very different from someone with a sore throat,
and no converter exists yet. They become useful once someone adds organ modules
for sepsis or pneumonia. PhysioNet restricts sending this data to third parties,
so keep it away from hosted AI services unless you have verified their rules.

## The data format

Each source folder holds three CSV files. Time zero is the first treatment.

### subjects.csv

| Column | Required | Meaning |
|---|---|---|
| subject_id | yes | Unique within the folder |
| study_id | yes | Keeps a whole study on one side of the train and validation split |
| kind | no | `individual`, or `arm` for a reported group average |
| conditions | no | Illness at time zero, such as `strep_a:2` for two days after symptoms began |
| split | no | `auto`, `train`, `val`, or `holdout` |
| age_years, sex, weight_kg, height_cm, creatinine_mg_dl, and others | no | Anything missing is sampled from similar people |
| notes | no | Where the numbers came from. Please always fill this in. |

### events.csv

| Column | Meaning |
|---|---|
| subject_id | Matches subjects.csv |
| time_h | Hours from time zero |
| type | `dose` or `exposure` |
| drug, dose_mg, route | For doses. Route is `oral` or `iv`. |
| pathogen | For exposures, such as `strep_a` |

### observations.csv

| Column | Meaning |
|---|---|
| subject_id | Matches subjects.csv |
| time_h | Hours from time zero |
| variable | See the list below |
| value | The measurement |
| sd | Optional. For a group average, the uncertainty of that average. |

Variables are `temperature_c`, `heart_rate_bpm`, `respiratory_rate_bpm`,
`throat_pain_0_10`, `cough_0_10`, `fatigue_0_10`, `diarrhea_0_10`, `wbc_k_ul`,
`crp_mg_l`, `throat_culture_log10_cfu`, `toxicity_index_0_1`,
`sore_throat_fraction`, `fever_fraction`, `culture_positive_fraction`, and
`conc_<drug>_mg_l`.

The fractions are shares of people, for trial outcomes such as "40% still had a
sore throat on day 3".

### A worked example

A trial reports that 30% of people on penicillin still had a positive culture on
day 2. Add this to `data/observations/custom/`.

subjects.csv:
```
subject_id,study_id,kind,conditions,split,notes
smith_pen,smith_2020,arm,strep_a:2,train,"Smith 2020 table 3"
```

events.csv, one row per dose:
```
subject_id,time_h,type,drug,dose_mg,route,pathogen
smith_pen,0,dose,penicillin_v,500,oral,
smith_pen,8,dose,penicillin_v,500,oral,
```

observations.csv:
```
subject_id,time_h,variable,value,sd
smith_pen,48,culture_positive_fraction,0.30,0.05
```

## Training commands

```powershell
# A first real run.
humanmock train --out checkpoints/run1

# Faster settings while experimenting.
humanmock train --out checkpoints/try --calibration-steps 100 --neural-steps 100

# Only physiological constants, with no neural training.
humanmock train --out checkpoints/calibrated --neural-steps 0

# An ensemble, so disagreement between models widens the reported spread.
humanmock train --out checkpoints/ensemble --members 3
```

To load a model, pass its folder to `humanmock serve --checkpoint`.

### Time and memory

No full training run has been timed yet. Compiling the model takes minutes before the
first progress line appears, so a quiet start is normal. Each stage compiles again.

Training uses a lot of memory, and the operating system may stop it if memory runs
out. Close games and other heavy programs first. If it still runs out, lower these:

```powershell
humanmock train --out checkpoints/run1 --batch-size 4 --people 4
```

Lower `--people` makes the shares of people in trial outcomes noisier, so avoid going
below 4 when training on those outcomes.

## Testing against unseen evidence

A score only means something when it comes from data the model never saw.

Splits always keep whole studies together. The hardest honest test removes a whole
study or a whole drug from training, then checks the prediction.

```powershell
humanmock train --out checkpoints/test --holdout-study culture_review_2023
humanmock train --out checkpoints/test --holdout-drug ibuprofen
humanmock evaluate checkpoints/test
```

The results compare the untrained and trained body in each measurement's own units.

## Checking the pipeline can learn at all

Synthetic data comes from a body with hidden laws the equations do not contain. It
tells you whether the pipeline can discover missing biology. It is never evidence
about real people.

```powershell
humanmock data synthetic --subjects 300
humanmock train --sources synthetic --out checkpoints/practice --people 1
humanmock evaluate checkpoints/practice
```

## Where to train

| Machine | When to use it |
|---|---|
| Your laptop CPU | The default, and enough for the current small body and small datasets |
| Linux or WSL2 with a GPU | Large cohorts. JAX has no NVIDIA support on native Windows, and WSL2 support is experimental. |
| Cloud Linux GPU | Long runs with several ensemble members |

To use a GPU on Linux or WSL2, install the CUDA build of JAX:

```bash
pip install -U "jax[cuda12]"
```

Small differential equation models often run as fast on a CPU, so measure before
paying for a GPU.

## What to expect

Be careful about what a good score means.

- **The current data is small.** A few studies cannot teach a network the human body.
  With this little real data, calibration does most of the useful work.
- **Neural training can overfit small data.** Watch the validation score, because
  training stops when it stops improving.
- **Synthetic success is a pipeline check.** It shows the method works in principle.
- **The body covers one illness.** Other illnesses need new organ modules and data.
  See CONTRIBUTING.md.
