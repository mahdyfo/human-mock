"""Generates practice data from a body with hidden laws the equations do not know.

This data is NOT evidence about real humans. It exists to answer one question:
can the training pipeline discover biology that is missing from the equations?
Because the hidden laws are known, the answer can be checked honestly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from humanmock.engine import DrugBook, Simulator
from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE
from humanmock.model import LAYOUT, OBS_INDEX, BodyModel

class HiddenTruthBody(BodyModel):
    """A body with extra laws. Neither the equations nor the networks are told about them."""

    def vector_field(self, t, y, c):
        d = super().vector_field(t, y, c)
        s = LAYOUT.unpack(y)
        # Hidden law 1: older people heal throat tissue more slowly.
        age_excess = jnp.maximum(c.patient["age_years"] - 35.0, 0.0) / 40.0
        tissue_i = LAYOUT.by_name["tissue"][1]
        d = d.at[tissue_i].add(0.012 * age_excess * jnp.maximum(s["tissue"], 0.0))
        # Hidden law 2: a sustained fever raises heart rate drive, seen through cytokine turnover.
        cytokine_i = LAYOUT.by_name["cytokine"][1]
        d = d.at[cytokine_i].add(-0.06 * jnp.maximum(s["temperature"] - 38.0, 0.0) * s["cytokine"])
        # Hidden law 3: paracetamol clears more slowly with age than the equations assume.
        conc_i = LAYOUT.by_name["conc"][1] + KNOWLEDGE.drug_slot("paracetamol")
        d = d.at[conc_i].add(0.15 * age_excess * jnp.maximum(s["conc"][KNOWLEDGE.drug_slot("paracetamol")], 0.0))
        return d


def hidden_truth_model(key) -> HiddenTruthBody:
    base = BodyModel.create(key)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    # The true constants also differ from the textbook ones.
    change_sys = {"k_heal": np.log(1.5), "pain_tissue": np.log(0.75), "k_crp_prod": np.log(0.8), "fever_max": np.log(0.9)}
    log_sys = np.array(fields["log_sys"])
    for name, value in change_sys.items():
        log_sys[KNOWLEDGE.sys_names.index(name)] = value
    fields["log_sys"] = jnp.asarray(log_sys)
    log_drug = np.array(fields["log_drug"])
    log_drug[KNOWLEDGE.drug_slot("paracetamol"), DRUG_FIELDS.index("cl_ref_l_h")] = np.log(1.25)
    fields["log_drug"] = jnp.asarray(log_drug)
    return HiddenTruthBody(**fields)


NOISE = {
    "temperature_c": 0.15,
    "heart_rate_bpm": 4.0,
    "throat_pain_0_10": 0.6,
    "crp_mg_l": 4.0,
    "wbc_k_ul": 0.8,
    "conc_paracetamol": 1.0,
    "conc_amoxicillin": 0.5,
}


def generate(out_dir: Path, subjects: int = 300, seed: int = 0, val_fraction: float = 0.25) -> dict:
    """Writes synthetic subjects in the canonical format."""
    rng = np.random.default_rng(seed)
    truth = hidden_truth_model(jax.random.PRNGKey(seed + 1))
    sim = Simulator(members=[truth], meta={"status": "synthetic_truth", "model_id": "hidden-truth"})
    book = DrugBook.build()
    rows_s, rows_e, rows_o = [], [], []
    for i in range(subjects):
        sid = f"syn_{i:05d}"
        age = float(rng.uniform(18, 80))
        sex = "female" if rng.random() < 0.5 else "male"
        weight = float(np.clip(rng.normal(77, 15), 45, 130))
        scenario = rng.choice(["paracetamol_pk", "amoxicillin_pk", "strep_untreated", "strep_treated", "viral"])
        conditions, onset = [], ""
        if scenario.startswith("strep"):
            conditions = [{"name": "strep_a", "days_since_symptom_onset": 1.0}]
            onset = "strep_a:1"
        elif scenario == "viral":
            conditions = [{"name": "viral_pharyngitis", "days_since_symptom_onset": 1.0}]
            onset = "viral_pharyngitis:1"
        patient = {"age_years": age, "sex": sex, "weight_kg": weight}
        p, _ = sim.create_particles(patient, conditions, book, 1, int(rng.integers(2**31)))

        doses: list[tuple[float, str, float]] = []
        if scenario == "paracetamol_pk":
            doses = [(0.0, "paracetamol", float(rng.choice([500, 1000, 1500])))]
            times = [0.5, 1, 2, 3, 4, 6, 8, 12]
            variables = ["conc_paracetamol"]
        elif scenario == "amoxicillin_pk":
            doses = [(0.0, "amoxicillin", float(rng.choice([250, 500, 875])))]
            times = [0.5, 1, 1.5, 2, 3, 4, 6]
            variables = ["conc_amoxicillin"]
        else:
            if scenario == "strep_treated":
                doses = [(k * 8.0, "amoxicillin", 500.0) for k in range(int(rng.choice([9, 15, 30])))]
            times = [12, 24, 48, 72, 96, 120, 168, 240]
            variables = ["temperature_c", "heart_rate_bpm", "throat_pain_0_10", "crp_mg_l", "wbc_k_ul"]

        rows_s.append(
            {
                "subject_id": sid,
                "study_id": f"synthetic:{'val' if i < subjects * val_fraction else 'train'}",
                "kind": "individual",
                "n": 1,
                "age_years": age,
                "sex": sex,
                "weight_kg": weight,
                "conditions": onset,
                "split": "val" if i < subjects * val_fraction else "train",
                "notes": "SYNTHETIC, not evidence about real humans",
            }
        )
        for at, drug, mg in doses:
            rows_e.append({"subject_id": sid, "time_h": at, "type": "dose", "drug": drug, "dose_mg": mg, "route": "oral"})
        t = 0.0
        for now in times:
            window = [
                {"type": "dose", "drug": drug, "dose_mg": mg, "at_hours": at - t}
                for at, drug, mg in doses
                if t <= at < now
            ]
            p = sim.advance(p, book, now - t, window)
            t = float(now)
            obs = sim.observe(p, book)[0]
            for variable in variables:
                if variable.startswith("conc_"):
                    drug = variable[5:]
                    value = obs[OBS_INDEX[f"conc_slot{KNOWLEDGE.drug_slot(drug)}"]]
                    name = f"conc_{drug}_mg_l"
                else:
                    value = obs[OBS_INDEX[variable]]
                    name = variable
                measured = float(value + rng.normal(0, NOISE[variable]))
                rows_o.append({"subject_id": sid, "time_h": now, "variable": name, "value": max(measured, 0.0)})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_s).to_csv(out_dir / "subjects.csv", index=False)
    pd.DataFrame(rows_e, columns=["subject_id", "time_h", "type", "drug", "dose_mg", "route", "pathogen"]).to_csv(
        out_dir / "events.csv", index=False
    )
    pd.DataFrame(rows_o).to_csv(out_dir / "observations.csv", index=False)
    return {"subjects": subjects, "observations": len(rows_o)}

