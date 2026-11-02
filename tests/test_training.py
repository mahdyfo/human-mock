"""Data formats, checkpoints, and proof that training really learns."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from humanmock.api import create_app
from humanmock.checkpoint import load_simulator, save_checkpoint, structure_fingerprint
from humanmock.data.formats import (
    DataFormatError,
    assign_splits,
    check_source,
    load_records,
    read_source,
    sample_people,
    variable_index,
)
from humanmock.data.synthetic import generate, hidden_truth_model
from humanmock.engine import DrugBook, Simulator
from humanmock.knowledge import KNOWLEDGE
from humanmock.model import OBS_INDEX, TRAIN_SOLVER, BodyModel
from humanmock.organs import ORGANS
from humanmock.training.train import (
    Batcher,
    TrainConfig,
    _run_stage,
    _stage_filter,
    _train_step,
    per_variable_error,
    predict_subject,
    run_training,
)

LITERATURE = Path(__file__).parent.parent / "data" / "observations"


def slot_name(drug: str) -> str:
    return f"conc_slot{KNOWLEDGE.drug_slot(drug)}"


def write_source(folder: Path, subjects: list[dict], events: list[dict], observations: list[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subjects).to_csv(folder / "subjects.csv", index=False)
    pd.DataFrame(events, columns=["subject_id", "time_h", "type", "drug", "dose_mg", "route", "pathogen"]).to_csv(
        folder / "events.csv", index=False
    )
    pd.DataFrame(observations).to_csv(folder / "observations.csv", index=False)
    return folder


def test_a_custom_csv_source_is_read_correctly(tmp_path):
    folder = write_source(
        tmp_path / "mine",
        [{"subject_id": "a", "study_id": "s1", "kind": "individual", "age_years": 40, "sex": "female", "conditions": "strep_a:2"}],
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "amoxicillin", "dose_mg": 500, "route": "oral"}],
        [{"subject_id": "a", "time_h": 2, "variable": "conc_amoxicillin_mg_l", "value": 5.0, "sd": 1.0}],
    )
    record = read_source(folder)[0]
    assert record.known == {"age_years": 40.0, "sex": "female"}
    assert record.onset_days == 2.0 and record.path_slot == KNOWLEDGE.path_names.index("strep_a")
    assert list(record.times) == [0.0, 2.0] and list(record.dt) == [0.0, 2.0]
    assert record.oral[0, KNOWLEDGE.drug_slot("amoxicillin")] == 500.0
    assert record.obs_var[0] == OBS_INDEX[slot_name("amoxicillin")]


@pytest.mark.parametrize(
    "bad_subject,bad_event,bad_obs,message",
    [
        ({"conditions": "plague:1"}, {}, {}, "unknown condition"),
        ({}, {"drug": "magic"}, {}, "unknown drug"),
        ({}, {}, {"variable": "mood"}, "Unknown variable"),
        ({"split": "banana"}, {}, {}, "unknown split"),
    ],
)
def test_bad_data_is_rejected_with_a_clear_message(tmp_path, bad_subject, bad_event, bad_obs, message):
    folder = write_source(
        tmp_path / "bad",
        [{"subject_id": "a", "study_id": "s", **bad_subject}],
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "amoxicillin", "dose_mg": 500, **bad_event}],
        [{"subject_id": "a", "time_h": 1, "variable": "temperature_c", "value": 37.0, **bad_obs}],
    )
    with pytest.raises(DataFormatError, match=message):
        read_source(folder)


@pytest.mark.parametrize(
    "bad_subject,bad_event,bad_obs,message",
    [
        ({}, {"subject_id": "ghost"}, {}, "not in subjects.csv"),
        ({}, {}, {"subject_id": "ghost"}, "not in subjects.csv"),
        ({}, {"time_h": -1}, {}, "time_h of 0 or more"),
        ({}, {}, {"time_h": "soon"}, "time_h of 0 or more"),
        ({}, {"dose_mg": None}, {}, "needs a dose_mg above 0"),
        ({}, {"dose_mg": -5}, {}, "needs a dose_mg above 0"),
        ({}, {}, {"value": "high"}, "numeric value"),
        ({}, {}, {"variable": "fever_fraction", "value": 40}, "between 0 and 1"),
        ({}, {}, {"sd": -1}, "negative sd"),
    ],
)
def test_rows_that_would_corrupt_training_are_refused(tmp_path, bad_subject, bad_event, bad_obs, message):
    folder = write_source(
        tmp_path / "bad",
        [{"subject_id": "a", "study_id": "s", **bad_subject}],
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "amoxicillin", "dose_mg": 500, **bad_event}],
        [{"subject_id": "a", "time_h": 1, "variable": "temperature_c", "value": 37.0, **bad_obs}],
    )
    with pytest.raises(DataFormatError, match=message):
        read_source(folder)
    assert any(message in error for error in check_source(folder).errors)


def test_the_data_check_explains_incomplete_and_empty_folders(tmp_path):
    (tmp_path / "empty").mkdir()
    assert check_source(tmp_path / "empty").errors == []
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "subjects.csv").write_text("subject_id,study_id\na,s\n", encoding="utf8")
    assert "events.csv" in check_source(partial).errors[0]


def test_the_data_check_notes_what_is_worth_fixing(tmp_path):
    folder = write_source(
        tmp_path / "thin",
        [{"subject_id": "a", "study_id": "s", "kind": "arm"}, {"subject_id": "b", "study_id": "s", "kind": "arm"}],
        [],
        [{"subject_id": "a", "time_h": 1, "variable": "fever_fraction", "value": 0.4}],
    )
    report = check_source(folder)
    text = " ".join(report.notes)
    assert report.errors == [] and report.subjects == 1 and report.measurements == 1
    assert "no measurements" in text and "empty notes" in text and "no sd" in text


def test_every_committed_data_source_passes_the_check():
    for folder in sorted(p for p in LITERATURE.iterdir() if p.is_dir()):
        assert check_source(folder).errors == [], folder


def test_variable_names_cannot_reach_internal_slots():
    with pytest.raises(DataFormatError):
        variable_index("conc_slot0")
    assert variable_index("conc_paracetamol_mg_l") == OBS_INDEX[slot_name("paracetamol")]


def test_splits_keep_whole_studies_together(tmp_path):
    subjects = [{"subject_id": f"p{i}", "study_id": f"s{i % 5}", "split": "auto"} for i in range(30)]
    observations = [{"subject_id": f"p{i}", "time_h": 1, "variable": "temperature_c", "value": 37.0} for i in range(30)]
    write_source(tmp_path / "many", subjects, [], observations)
    records = load_records(tmp_path)
    assign_splits(records, seed=1, val_fraction=0.2)
    by_study: dict[str, set] = {}
    for record in records:
        by_study.setdefault(record.study_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_study.values())
    assert {"train", "val"} == {r.split for r in records}


def test_holding_out_a_drug_removes_every_subject_given_it(tmp_path):
    write_source(
        tmp_path / "two",
        [{"subject_id": "a", "study_id": "s1", "split": "train"}, {"subject_id": "b", "study_id": "s2", "split": "train"}],
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "ibuprofen", "dose_mg": 200}],
        [
            {"subject_id": "a", "time_h": 1, "variable": "temperature_c", "value": 37.0},
            {"subject_id": "b", "time_h": 1, "variable": "temperature_c", "value": 37.0},
        ],
    )
    records = load_records(tmp_path)
    assign_splits(records, seed=0, holdout_drugs=["ibuprofen"])
    assert {r.subject_id: r.split for r in records} == {"a": "holdout", "b": "train"}


def test_synthetic_data_stays_out_unless_requested(tmp_path):
    for name in ("real", "synthetic"):
        write_source(
            tmp_path / name,
            [{"subject_id": f"{name}1", "study_id": name}],
            [],
            [{"subject_id": f"{name}1", "time_h": 1, "variable": "temperature_c", "value": 37.0}],
        )
    assert {r.source for r in load_records(tmp_path)} == {"real"}
    assert {r.source for r in load_records(tmp_path, include_synthetic=True)} == {"real", "synthetic"}


def test_the_shipped_literature_data_loads():
    records = read_source(LITERATURE / "literature")
    assert len(records) >= 8
    assert any(r.split == "val" for r in records)


def test_batches_cover_every_subject_and_mask_filler(tmp_path):
    subjects = [{"subject_id": f"p{i}", "study_id": "s"} for i in range(5)]
    observations = [{"subject_id": f"p{i}", "time_h": 1, "variable": "temperature_c", "value": 37.0} for i in range(5)]
    write_source(tmp_path / "x", subjects, [], observations)
    records = load_records(tmp_path)
    sample_people(records, 2, 0)
    batches = list(Batcher(records, 4, 0).all_batches())
    assert sum(float(b["mask"].sum()) for b in batches) == 5.0


def test_a_small_group_never_repeats_a_subject_to_fill_a_batch(tmp_path):
    subjects = [{"subject_id": f"p{i}", "study_id": "s"} for i in range(3)]
    observations = [{"subject_id": f"p{i}", "time_h": 1, "variable": "temperature_c", "value": 37.0 + i} for i in range(3)]
    write_source(tmp_path / "x", subjects, [], observations)
    records = load_records(tmp_path)
    sample_people(records, 2, 0)
    batch = Batcher(records, 16, 0).sample()
    assert batch["dt"].shape[0] == 3
    assert sorted(float(v) for v in batch["obs_value"][:, 0]) == [37.0, 38.0, 39.0]


def test_scores_name_drugs_rather_than_internal_slots():
    from humanmock.training.train import _readable

    assert _readable(slot_name("amoxicillin")) == "conc_amoxicillin_mg_l"
    assert _readable("temperature_c") == "temperature_c"


def test_untrained_model_predicts_a_known_amoxicillin_curve(tmp_path):
    write_source(
        tmp_path / "x",
        [{"subject_id": "a", "study_id": "s", "age_years": 30, "weight_kg": 70}],
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "amoxicillin", "dose_mg": 500, "route": "oral"}],
        [{"subject_id": "a", "time_h": 1.5, "variable": "conc_amoxicillin_mg_l", "value": 6.5}],
    )
    records = load_records(tmp_path)
    sample_people(records, 4, 0)
    batch = {k: v[0] for k, v in Batcher(records, 1, 0).collate(records, 8, 8).items()}
    model = BodyModel.create(jax.random.PRNGKey(0))
    prediction, ok = predict_subject(model, batch, jax.random.PRNGKey(1), TRAIN_SOLVER)
    assert float(ok) == 1.0
    assert 4.5 < float(prediction[0]) < 8.5


def test_checkpoint_round_trip_keeps_results_identical(tmp_path):
    model = BodyModel.create(jax.random.PRNGKey(5))
    changed = eqx.tree_at(lambda m: m.log_sys, model, model.log_sys + 0.1)
    save_checkpoint(tmp_path / "ckpt", [changed], {"support": {}, "data": {}, "scores": {}})
    sim = load_simulator(tmp_path / "ckpt")
    assert sim.meta["status"] == "trained"
    assert sim.meta["structure"] == structure_fingerprint()
    assert np.allclose(np.asarray(sim.members[0].log_sys), np.asarray(changed.log_sys))


def test_a_checkpoint_from_a_different_body_is_refused(tmp_path):
    import json

    model = BodyModel.create(jax.random.PRNGKey(5))
    save_checkpoint(tmp_path / "ckpt", [model], {"support": {}, "data": {}, "scores": {}})
    meta_path = tmp_path / "ckpt" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["structure"] = "something-else"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different body structure"):
        load_simulator(tmp_path / "ckpt")


def test_sessions_from_an_old_model_ask_to_reset(tmp_path):
    untrained = TestClient(create_app(load_or_prior(None), tmp_path / "s.db"))
    sid = untrained.post("/v1/sessions", json={"people": 4, "seed": 1}).json()["session_id"]
    model = BodyModel.create(jax.random.PRNGKey(5))
    save_checkpoint(tmp_path / "ckpt", [model], {"support": {"conditions": [], "drugs": {}, "covariates": {}}, "data": {}, "scores": {}})
    trained = TestClient(create_app(load_simulator(tmp_path / "ckpt"), tmp_path / "s.db"))
    assert trained.post(f"/v1/sessions/{sid}/step", json={"duration": "1h"}).status_code == 409
    assert trained.post(f"/v1/sessions/{sid}/reset", json={"new_people": True}).status_code == 200
    assert trained.post(f"/v1/sessions/{sid}/step", json={"duration": "1h"}).status_code == 200


def load_or_prior(_):
    return Simulator()


def test_trained_model_reports_when_a_request_leaves_its_training_data(tmp_path):
    model = BodyModel.create(jax.random.PRNGKey(5))
    support = {
        "conditions": ["strep_a"],
        "drugs": {"amoxicillin": (250.0, 875.0)},
        "covariates": {"age_years": (20.0, 45.0)},
    }
    save_checkpoint(tmp_path / "ckpt", [model], {"support": support, "data": {}, "scores": {}})
    client = TestClient(create_app(load_simulator(tmp_path / "ckpt"), tmp_path / "s.db"))
    body = {"patient": {"age_years": 30}, "conditions": [{"name": "strep_a"}], "people": 8, "seed": 1}
    created = client.post("/v1/sessions", json=body).json()
    assert created["model"]["status"] == "trained"
    sid = created["session_id"]
    inside = client.post(
        f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{"drug": "amoxicillin", "dose_mg": 500}]}
    ).json()
    assert inside["support"]["within_training_support"] is True
    outside = client.post(
        f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{"drug": "ibuprofen", "dose_mg": 400}]}
    ).json()
    assert outside["support"]["within_training_support"] is False
    assert any("ibuprofen" in reason for reason in outside["support"]["reasons"])
    older = client.post("/v1/sessions", json={**body, "patient": {"age_years": 70}}).json()
    assert older["support"]["within_training_support"] is False


def _long_illness(tmp_path):
    """Someone ill for a long stretch without any drug, which once froze all training."""
    write_source(
        tmp_path / "ill",
        [{"subject_id": "a", "study_id": "s", "age_years": 40, "weight_kg": 70, "conditions": "viral_pharyngitis:1"}],
        [],
        [{"subject_id": "a", "time_h": t, "variable": "throat_pain_0_10", "value": 3.0} for t in (24, 96, 240)],
    )
    records = load_records(tmp_path)
    sample_people(records, 4, 0)
    return records


def test_gradients_stay_finite_through_a_long_illness_without_drugs(tmp_path):
    """An undosed drug is a fast state at exactly zero. Its gradient must not overflow."""
    records = _long_illness(tmp_path)
    batch = {k: v[0] for k, v in Batcher(records, 1, 0).collate(records, 8, 8).items()}
    model = BodyModel.create(jax.random.PRNGKey(0))

    def loss(m):
        prediction, _ = predict_subject(m, batch, jax.random.PRNGKey(1), TRAIN_SOLVER)
        return jnp.sum(prediction * batch["mask"])

    gradients = eqx.filter_jit(eqx.filter_grad(loss))(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradients, eqx.is_inexact_array))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    # The neural cells must really receive a signal, or they can never learn.
    cell_size = sum(float(jnp.abs(g).sum()) for g in jax.tree_util.tree_leaves(eqx.filter(gradients.cells, eqx.is_inexact_array)))
    assert cell_size > 0.0


def test_a_training_step_really_moves_the_neural_cells(tmp_path):
    records = _long_illness(tmp_path)
    batch = Batcher(records, 1, 0).collate(records, 8, 8)
    model = BodyModel.create(jax.random.PRNGKey(0))
    diff, static = eqx.partition(model, _stage_filter(model, "neural"))
    optimizer = optax.apply_if_finite(optax.adam(0.002), max_consecutive_errors=25)
    state = optimizer.init(diff)
    config = TrainConfig()
    new_diff, state, _, _ = _train_step(diff, static, state, batch, jax.random.PRNGKey(1), optimizer, config, TRAIN_SOLVER)
    assert int(state.total_notfinite) == 0
    moved = {o.name: float(jnp.abs(c.layers[-1].weight).max()) for o, c in zip(ORGANS, new_diff.cells)}
    # This person only reports throat pain, so exactly the organs that drive pain should learn.
    for organ in ("pathogen", "immunity", "inflammation", "throat"):
        assert moved[organ] > 1e-4, moved


def test_the_training_command_runs_from_data_to_a_served_model(tmp_path):
    """The whole path a contributor uses, kept tiny so that it stays quick."""
    data = tmp_path / "observations"
    write_source(
        data / "mine",
        [{"subject_id": f"p{i}", "study_id": f"s{i}", "age_years": 40, "weight_kg": 70} for i in range(4)],
        [{"subject_id": f"p{i}", "time_h": 0, "type": "dose", "drug": "paracetamol", "dose_mg": 1000, "route": "oral"} for i in range(4)],
        [{"subject_id": f"p{i}", "time_h": 2, "variable": "conc_paracetamol_mg_l", "value": 9.0} for i in range(4)],
    )
    config = TrainConfig(calibration_steps=1, neural_steps=1, people=2, batch_size=2, eval_every=1)
    meta = run_training(data, tmp_path / "ckpt", config, log=lambda _: None)
    assert "conc_paracetamol_mg_l" in meta["scores"]["train"]["trained"]
    assert meta["history"][0][-1]["skipped"] == 0
    sim = load_simulator(tmp_path / "ckpt")
    book = DrugBook.build()
    p, _ = sim.create_particles({"age_years": 30}, [], book, 2, 1)
    assert np.isfinite(sim.observe(sim.advance(p, book, 1.0, []), book)).all()


def _how_much_higher_in_the_old(model) -> float:
    """Paracetamol level in a 75 year old compared with a 25 year old, 6 h after a dose."""
    sim = Simulator(members=[model], meta={"status": "trained", "model_id": "t"})
    book = DrugBook.build()
    levels = []
    for age in (25, 75):
        person = {"age_years": age, "sex": "male", "weight_kg": 75, "creatinine_mg_dl": 0.9}
        p, _ = sim.create_particles(person, [], book, 32, 1)
        p = sim.advance(p, book, 6.0, [{"type": "dose", "drug": "paracetamol", "dose_mg": 1000, "at_hours": 0}])
        levels.append(float(np.median(sim.observe(p, book)[:, OBS_INDEX[slot_name("paracetamol")]])))
    return levels[1] / levels[0]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [3, 11])
def test_neurons_discover_a_law_the_equations_do_not_contain(tmp_path, seed):
    """The real test of the whole idea.

    In the practice data, older people clear paracetamol more slowly. No equation in
    the body says so, and no single constant can express it, because it depends on age.
    Calibration alone must therefore fall short, and only the neural parts can close
    the gap, on people they never trained on.
    """
    generate(tmp_path / "synthetic", subjects=72, seed=seed, scenarios=("paracetamol_pk",))
    records = load_records(tmp_path, include_synthetic=True)
    sample_people(records, 32, 0)
    assign_splits(records, 0)
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    config = TrainConfig(batch_size=8, people=4, eval_every=25, patience=16)
    key, k_cal, k_neural = jax.random.split(jax.random.PRNGKey(seed), 3)
    quiet = lambda _: None  # noqa: E731

    prior = BodyModel.create(jax.random.PRNGKey(0))
    calibrated, h1 = _run_stage(prior, "calibration", 150, 0.02, train, val, config, TRAIN_SOLVER, k_cal, quiet)
    neural, h2 = _run_stage(calibrated, "neural", 400, 0.002, train, val, config, TRAIN_SOLVER, k_neural, quiet)
    truth = hidden_truth_model(jax.random.PRNGKey(seed + 1), 0.2)

    def error(model):
        return per_variable_error(model, val, 8, key, TRAIN_SOLVER, 4)["conc_paracetamol_mg_l"]["rmse"]

    assert all(row.get("skipped", 0) == 0 for row in h1 + h2), "Training skipped steps."
    assert error(neural) < 0.95 * error(calibrated), (error(prior), error(calibrated), error(neural), error(truth))

    # The law itself must appear, rather than only a better score.
    found, without, actual = (_how_much_higher_in_the_old(m) for m in (neural, calibrated, truth))
    assert without + 0.3 * (actual - without) < found, (without, found, actual)

    # The trained body must still obey basic physics when served.
    sim = Simulator(members=[neural], meta={"status": "trained", "model_id": "t"})
    book = DrugBook.build()
    p, _ = sim.create_particles({"age_years": 30, "weight_kg": 70}, [], book, 4, 1)
    later = sim.observe(sim.advance(p, book, 240.0, []), book)
    assert np.isfinite(later).all()
    assert later[:, OBS_INDEX[slot_name("amoxicillin")]].max() == 0.0
