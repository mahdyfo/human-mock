"""Prints how the simulator behaves in standard scenarios, next to what is known.

Run this after changing physiology or training a model. It is the quickest way
to see whether the body still behaves sensibly.

    python scripts/behaviour_report.py [--checkpoint checkpoints/run1] [--people 64]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from humanmock.engine import DrugBook, Simulator
from humanmock.model import OBS_INDEX

ADULT = {"age_years": 30, "sex": "male", "weight_kg": 75}
DAYS = (1, 2, 3, 5, 7, 10, 14, 21)


def course(sim, book, patient, conditions, doses, n, days=DAYS, seed=1):
    """doses is a list of (drug, mg, every_h, total_doses). Time zero is treatment start."""
    p, _ = sim.create_particles(patient, conditions, book, n, seed)
    events = []
    for drug, mg, every, count in doses:
        events.extend({"type": "dose", "drug": drug, "dose_mg": mg, "at_hours": k * every} for k in range(count))
    rows, t = {}, 0.0
    obs0 = sim.observe(p, book)
    rows[0] = obs0
    for day in days:
        end = day * 24.0
        window = [dict(e, at_hours=e["at_hours"] - t) for e in events if t <= e["at_hours"] < end]
        p = sim.advance(p, book, end - t, window)
        t = end
        rows[day] = sim.observe(p, book)
    return rows, p


def line(rows, name, stat="median"):
    out = []
    for day, obs in rows.items():
        v = obs[:, OBS_INDEX[name]]
        out.append(f"d{day}:{np.nanmedian(v):.1f}" if stat == "median" else f"d{day}:{np.nanmean(v):.2f}")
    return " ".join(out)


def share(rows, name, thr):
    return " ".join(f"d{d}:{(o[:, OBS_INDEX[name]] > thr).mean():.2f}" for d, o in rows.items())


def run(sim: Simulator, n: int) -> None:
    book = DrugBook.build()
    strep = [{"name": "strep_a", "days_since_symptom_onset": 1.0}]
    viral = [{"name": "viral_pharyngitis", "days_since_symptom_onset": 1.0}]

    print("\n== Healthy adult, nothing happens")
    rows, _ = course(sim, book, ADULT, [], [], n, days=(1, 7, 30))
    print(" temperature", line(rows, "temperature_c"), "| pain", line(rows, "throat_pain_0_10"))

    print("\n== Untreated strep throat")
    untreated, _ = course(sim, book, ADULT, strep, [], n)
    print(" pain       ", line(untreated, "throat_pain_0_10"))
    print(" temperature", line(untreated, "temperature_c"))
    print(" sore throat", share(untreated, "throat_pain_0_10", 2.0), " target about 0.57 at d3 and 0.18 at d7")
    print(" culture+   ", share(untreated, "throat_culture_log10_cfu", 2.0))
    print(" cough      ", line(untreated, "cough_0_10"), "| CRP", line(untreated, "crp_mg_l"), "| WBC", line(untreated, "wbc_k_ul"))

    print("\n== Strep throat, amoxicillin 500 mg every 8 h for 10 days")
    full, _ = course(sim, book, ADULT, strep, [("amoxicillin", 500, 8, 30)], n)
    print(" pain       ", line(full, "throat_pain_0_10"))
    print(" temperature", line(full, "temperature_c"))
    print(" sore throat", share(full, "throat_pain_0_10", 2.0), " target about 0.40 at d3 and 0.09 at d7")
    print(" culture+   ", share(full, "throat_culture_log10_cfu", 2.0), " target about 0.07 d1, 0.05 d2, 0.03 d3 to 9")
    print(" diarrhea   ", line(full, "diarrhea_0_10"))

    print("\n== Strep throat, antibiotic stopped early after 3 days")
    short, _ = course(sim, book, ADULT, strep, [("amoxicillin", 500, 8, 9)], n)
    print(" culture+   ", share(short, "throat_culture_log10_cfu", 2.0))
    print(" sore throat", share(short, "throat_pain_0_10", 2.0))

    print("\n== Viral sore throat with and without amoxicillin")
    vnone, _ = course(sim, book, ADULT, viral, [], n)
    vabx, _ = course(sim, book, ADULT, viral, [("amoxicillin", 500, 8, 30)], n)
    print(" pain, no drug   ", line(vnone, "throat_pain_0_10"))
    print(" pain, antibiotic", line(vabx, "throat_pain_0_10"))
    print(" cough           ", line(vnone, "cough_0_10"))

    print("\n== First hours after one amoxicillin dose: symptoms should barely move at 1 h")
    p0, _ = sim.create_particles(ADULT, strep, book, n, 1)
    before = sim.observe(p0, book)
    p1 = sim.advance(p0, book, 1.0, [{"type": "dose", "drug": "amoxicillin", "dose_mg": 500, "at_hours": 0}])
    after = sim.observe(p1, book)
    for name in ("throat_pain_0_10", "temperature_c", "throat_culture_log10_cfu"):
        print(f" {name}: {np.nanmedian(before[:, OBS_INDEX[name]]):.2f} then {np.nanmedian(after[:, OBS_INDEX[name]]):.2f}")
    print(f" amoxicillin level at 1 h: {np.nanmedian(after[:, OBS_INDEX['conc_slot0']]):.2f} mg/L, label peak range 5.5 to 7.5")

    print("\n== Ibuprofen 400 mg during a fever: temperature should fall, then return")
    pf, _ = sim.create_particles(ADULT, strep, book, n, 1)
    now = sim.observe(pf, book)
    print(f" before {np.nanmedian(now[:, OBS_INDEX['temperature_c']]):.2f}", end="")
    pf = sim.advance(pf, book, 2.0, [{"type": "dose", "drug": "ibuprofen", "dose_mg": 400, "at_hours": 0}])
    print(f", 2 h {np.nanmedian(sim.observe(pf, book)[:, OBS_INDEX['temperature_c']]):.2f}", end="")
    pf = sim.advance(pf, book, 2.0, [])
    print(f", 4 h {np.nanmedian(sim.observe(pf, book)[:, OBS_INDEX['temperature_c']]):.2f}", end="")
    pf = sim.advance(pf, book, 8.0, [])
    print(f", 12 h {np.nanmedian(sim.observe(pf, book)[:, OBS_INDEX['temperature_c']]):.2f}")

    print("\n== Immune memory: second strep exposure after recovery should be milder")
    first, p_after = course(sim, book, ADULT, [], [], n, days=(30,))
    p_first = sim.advance(p_after, book, 30 * 24.0, [{"type": "exposure", "pathogen": "strep_a", "at_hours": 0}])
    # A fresh body and a previously infected body meet the same exposure.
    fresh = p_after
    peak_first, peak_second = [], []
    exposure = [{"type": "exposure", "pathogen": "strep_a", "at_hours": 0}]
    a = sim.advance(fresh, book, 24.0, exposure)
    b = sim.advance(p_first, book, 24.0, exposure)
    for _ in range(8):
        peak_first.append(sim.observe(a, book)[:, OBS_INDEX["throat_pain_0_10"]])
        peak_second.append(sim.observe(b, book)[:, OBS_INDEX["throat_pain_0_10"]])
        a = sim.advance(a, book, 24.0, [])
        b = sim.advance(b, book, 24.0, [])
    print(f" peak median pain, first infection {np.nanmedian(np.max(peak_first, axis=0)):.2f}, second {np.nanmedian(np.max(peak_second, axis=0)):.2f}")

    print("\n== Kidney impairment: same amoxicillin dose should leave higher levels")
    weak = dict(ADULT, age_years=78, creatinine_mg_dl=3.0)
    for label, patient in (("normal", ADULT), ("impaired", weak)):
        pk, _ = sim.create_particles(patient, [], book, n, 1)
        pk = sim.advance(pk, book, 6.0, [{"type": "dose", "drug": "amoxicillin", "dose_mg": 500, "at_hours": 0}])
        print(f" {label}: {np.nanmedian(sim.observe(pk, book)[:, OBS_INDEX['conc_slot0']]):.2f} mg/L at 6 h")

    print("\n== Overdose: paracetamol 15 g should raise organ stress far above 1 g")
    for grams in (1, 15):
        po, _ = sim.create_particles(ADULT, [], book, n, 1)
        po = sim.advance(po, book, 24.0, [{"type": "dose", "drug": "paracetamol", "dose_mg": grams * 1000, "at_hours": 0}])
        print(f" {grams} g: toxicity index {np.nanmedian(sim.observe(po, book)[:, OBS_INDEX['toxicity_index_0_1']]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--people", type=int, default=64)
    args = parser.parse_args()
    if args.checkpoint:
        from humanmock.checkpoint import load_simulator

        sim = load_simulator(args.checkpoint)
    else:
        sim = Simulator()
    started = time.time()
    run(sim, args.people)
    print(f"\nFinished in {time.time() - started:.0f} s with model '{sim.model_id}'.")


if __name__ == "__main__":
    main()
