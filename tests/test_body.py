"""Basic physics and the behaviours the body must show. Failing here means the body is wrong."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from humanmock.engine import DrugBook, Simulator
from humanmock.knowledge import KNOWLEDGE, MAX_DRUGS
from humanmock.model import LAYOUT, OBS_INDEX, OBS_NAMES, BodyModel

ADULT = {"age_years": 30, "sex": "male", "weight_kg": 75, "creatinine_mg_dl": 0.9}
STREP = [{"name": "strep_a", "days_since_symptom_onset": 1.0}]
AMOX = {"type": "dose", "drug": "amoxicillin", "dose_mg": 500, "at_hours": 0}


def median(sim, book, p, name):
    return float(np.nanmedian(sim.observe(p, book)[:, OBS_INDEX[name]]))


def share(sim, book, p, name, threshold):
    return float((sim.observe(p, book)[:, OBS_INDEX[name]] > threshold).mean())


def treat(sim, book, p, drug, mg, every, count, days):
    events = [{"type": "dose", "drug": drug, "dose_mg": mg, "at_hours": k * every} for k in range(count)]
    events = [e for e in events if e["at_hours"] < days * 24]
    return sim.advance(p, book, days * 24.0, events)


def test_every_neural_output_starts_at_zero():
    model = BodyModel.create(jax.random.PRNGKey(3))
    cov = jnp.array([30, 0, 75, 175, 0.9, 22, 70, 14, 7, 1, 36.7, 120, 75, 1, 1], dtype=jnp.float32)
    c = model.context(
        cov,
        jnp.zeros(len(KNOWLEDGE.sys_names)),
        jnp.zeros((KNOWLEDGE.n_pathogens, len(KNOWLEDGE.path_fields))),
        jnp.zeros((MAX_DRUGS, KNOWLEDGE.drug_table.shape[1])),
        jnp.asarray(KNOWLEDGE.drug_table),
    )
    assert np.allclose(np.asarray(c.embed), 0.0)
    y = model.initial_state(c)
    latent_rate = model.vector_field(0.0, y, c)[LAYOUT.physio_size :]
    assert np.allclose(np.asarray(latent_rate), 0.0)


def test_healthy_body_stays_healthy(sim, book):
    p, _ = sim.create_particles(ADULT, [], book, 16, 1)
    start = sim.observe(p, book)
    later = sim.observe(sim.advance(p, book, 24.0 * 30, []), book)
    assert np.allclose(start[:, OBS_INDEX["temperature_c"]], later[:, OBS_INDEX["temperature_c"]], atol=0.05)
    assert later[:, OBS_INDEX["throat_pain_0_10"]].max() < 0.5
    assert later[:, OBS_INDEX["throat_culture_log10_cfu"]].max() < 0.5


def test_no_drug_appears_without_a_dose(sim, book):
    p, _ = sim.create_particles(ADULT, STREP, book, 16, 1)
    later = sim.observe(sim.advance(p, book, 240.0, []), book)
    for slot in range(MAX_DRUGS):
        assert later[:, OBS_INDEX[f"conc_slot{slot}"]].max() == 0.0


def test_time_steps_are_consistent(sim, book):
    """Advancing 8 h once must match advancing 1 h eight times."""
    p, _ = sim.create_particles(ADULT, STREP, book, 8, 2)
    p = sim.advance(p, book, 1e-3, [AMOX])
    once = sim.advance(p, book, 8.0, [])
    stepped = p
    for _ in range(8):
        stepped = sim.advance(stepped, book, 1.0, [])
    a, b = sim.observe(once, book), sim.observe(stepped, book)
    for name in ("temperature_c", "throat_pain_0_10", "conc_slot0", "throat_culture_log10_cfu"):
        assert np.allclose(a[:, OBS_INDEX[name]], b[:, OBS_INDEX[name]], rtol=0.02, atol=0.02), name


def test_results_repeat_for_the_same_seed(sim, book):
    a, _ = sim.create_particles(ADULT, STREP, book, 8, 7)
    b, _ = sim.create_particles(ADULT, STREP, book, 8, 7)
    c, _ = sim.create_particles(ADULT, STREP, book, 8, 8)
    assert np.array_equal(a.y, b.y)
    assert not np.array_equal(a.y, c.y)


def test_amoxicillin_peak_is_within_the_label_range(sim, book):
    p, _ = sim.create_particles(ADULT, [], book, 32, 1)
    peaks = []
    current = sim.advance(p, book, 1.0, [AMOX])
    peaks.append(median(sim, book, current, "conc_slot0"))
    current = sim.advance(current, book, 0.5, [])
    peaks.append(median(sim, book, current, "conc_slot0"))
    current = sim.advance(current, book, 0.5, [])
    peaks.append(median(sim, book, current, "conc_slot0"))
    assert 5.0 <= max(peaks) <= 8.0  # The label reports 5.5 to 7.5 mg/L at 1 to 2 h.


def test_one_hour_after_a_dose_symptoms_have_not_moved(sim, book):
    p, _ = sim.create_particles(ADULT, STREP, book, 32, 1)
    after = sim.advance(p, book, 1.0, [AMOX])
    assert abs(median(sim, book, after, "throat_pain_0_10") - median(sim, book, p, "throat_pain_0_10")) < 0.3
    assert abs(median(sim, book, after, "temperature_c") - median(sim, book, p, "temperature_c")) < 0.2
    assert median(sim, book, after, "conc_slot0") > 3.0


def test_antibiotics_clear_bacteria_and_speed_recovery(sim, book):
    p, _ = sim.create_particles(ADULT, STREP, book, 48, 1)
    treated = treat(sim, book, p, "amoxicillin", 500, 8, 9, 3)
    untreated = sim.advance(p, book, 72.0, [])
    assert share(sim, book, treated, "throat_culture_log10_cfu", 2.0) < 0.15
    assert share(sim, book, untreated, "throat_culture_log10_cfu", 2.0) > 0.4
    assert median(sim, book, treated, "throat_pain_0_10") < median(sim, book, untreated, "throat_pain_0_10")
    assert median(sim, book, treated, "temperature_c") < median(sim, book, untreated, "temperature_c")


def test_antibiotics_do_nothing_for_a_virus(sim, book):
    p, _ = sim.create_particles(ADULT, [{"name": "viral_pharyngitis"}], book, 32, 1)
    treated = treat(sim, book, p, "amoxicillin", 500, 8, 9, 3)
    untreated = sim.advance(p, book, 72.0, [])
    assert abs(median(sim, book, treated, "throat_pain_0_10") - median(sim, book, untreated, "throat_pain_0_10")) < 0.1


def test_stopping_early_relapses_more_than_a_full_course(sim, book):
    n = 96
    p, _ = sim.create_particles(ADULT, STREP, book, n, 5)
    short = treat(sim, book, p, "amoxicillin", 500, 8, 9, 3)
    full = treat(sim, book, p, "amoxicillin", 500, 8, 30, 10)
    short_later = sim.advance(short, book, 24.0 * 4, [])
    full_later = treat(sim, book, p, "amoxicillin", 500, 8, 30, 7)
    assert share(sim, book, short_later, "throat_culture_log10_cfu", 2.0) > share(sim, book, full_later, "throat_culture_log10_cfu", 2.0)
    assert share(sim, book, full, "throat_culture_log10_cfu", 2.0) < 0.05


def test_fever_drug_lowers_temperature_then_wears_off(sim, book):
    p, _ = sim.create_particles(ADULT, STREP, book, 32, 1)
    start = median(sim, book, p, "temperature_c")
    two = sim.advance(p, book, 2.0, [{"type": "dose", "drug": "ibuprofen", "dose_mg": 400, "at_hours": 0}])
    twelve = sim.advance(two, book, 10.0, [])
    assert median(sim, book, two, "temperature_c") < start - 0.4
    assert median(sim, book, twelve, "temperature_c") > median(sim, book, two, "temperature_c") + 0.2


def test_immune_memory_makes_a_second_infection_milder(sim, book):
    exposure = [{"type": "exposure", "pathogen": "strep_a", "at_hours": 0}]
    p, _ = sim.create_particles(ADULT, [], book, 24, 3)
    first = sim.advance(p, book, 24.0, exposure)
    veteran = sim.advance(first, book, 24.0 * 40, [])
    second = sim.advance(veteran, book, 24.0, exposure)
    peak_first, peak_second = 0.0, 0.0
    a, b = first, second
    for _ in range(7):
        a = sim.advance(a, book, 24.0, [])
        b = sim.advance(b, book, 24.0, [])
        peak_first = max(peak_first, median(sim, book, a, "throat_pain_0_10"))
        peak_second = max(peak_second, median(sim, book, b, "throat_pain_0_10"))
    assert peak_second < peak_first * 0.5


def test_weak_kidneys_keep_a_renally_cleared_drug_longer(sim, book):
    normal, _ = sim.create_particles(ADULT, [], book, 16, 1)
    impaired, _ = sim.create_particles(dict(ADULT, age_years=78, creatinine_mg_dl=3.0), [], book, 16, 1)
    a = sim.advance(normal, book, 6.0, [AMOX])
    b = sim.advance(impaired, book, 6.0, [AMOX])
    assert median(sim, book, b, "conc_slot0") > 2 * median(sim, book, a, "conc_slot0")


def test_overdose_shows_organ_stress(sim, book):
    p, _ = sim.create_particles(ADULT, [], book, 16, 1)
    normal = sim.advance(p, book, 24.0, [{"type": "dose", "drug": "paracetamol", "dose_mg": 1000, "at_hours": 0}])
    overdose = sim.advance(p, book, 24.0, [{"type": "dose", "drug": "paracetamol", "dose_mg": 15000, "at_hours": 0}])
    assert median(sim, book, normal, "toxicity_index_0_1") < 0.01
    assert median(sim, book, overdose, "toxicity_index_0_1") > 0.1


def test_a_hypothetical_drug_acts_only_through_its_mechanism(sim):
    book = DrugBook.build(
        [
            {
                "name": "candidate_x",
                "ka": 1.0,
                "v_l_per_kg": 0.5,
                "cl_ref_l_h": 10.0,
                "kill_emax": 2.0,
                "kill_ec50": 0.05,
                "kill_hill": 2.0,
                "reservoir_penetration": 0.5,
            }
        ]
    )
    p, _ = sim.create_particles(ADULT, STREP, book, 32, 1)
    dosed = sim.advance(p, book, 48.0, [{"type": "dose", "drug": "candidate_x", "dose_mg": 500, "at_hours": 0}])
    nothing = sim.advance(p, book, 48.0, [])
    assert share(sim, book, dosed, "throat_culture_log10_cfu", 2.0) < share(sim, book, nothing, "throat_culture_log10_cfu", 2.0)


def test_missing_values_reflect_who_the_person_is(sim, book):
    _, older = sim.create_particles({"age_years": 75, "sex": "female"}, [], book, 512, 1)
    _, younger = sim.create_particles({"age_years": 25, "sex": "male"}, [], book, 512, 1)
    assert older["systolic_bp_mmhg"]["mean"] > younger["systolic_bp_mmhg"]["mean"] + 8
    assert older["height_cm"]["mean"] < younger["height_cm"]["mean"] - 8
    assert "age_years" not in older and "weight_kg" in older


def test_observable_names_match_what_organs_return():
    model = BodyModel.create(jax.random.PRNGKey(0))
    cov = jnp.array([30, 0, 75, 175, 0.9, 22, 70, 14, 7, 1, 36.7, 120, 75, 1, 1], dtype=jnp.float32)
    c = model.context(
        cov,
        jnp.zeros(len(KNOWLEDGE.sys_names)),
        jnp.zeros((KNOWLEDGE.n_pathogens, len(KNOWLEDGE.path_fields))),
        jnp.zeros((MAX_DRUGS, KNOWLEDGE.drug_table.shape[1])),
        jnp.asarray(KNOWLEDGE.drug_table),
    )
    assert model.observe(model.initial_state(c), c).shape == (len(OBS_NAMES),)
