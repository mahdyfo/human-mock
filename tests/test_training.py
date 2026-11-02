"""Data formats, checkpoints, and proof that training really learns."""

from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from humanmock.api import create_app
from humanmock.checkpoint import load_simulator, save_checkpoint, structure_fingerprint
from humanmock.data.formats import (
    DataFormatError,
    assign_splits,
    load_records,
    read_source,
    sample_people,
    variable_index,
)
from humanmock.data.synthetic import generate
from humanmock.engine import DrugBook
from humanmock.model import OBS_INDEX, TRAIN_SOLVER, BodyModel
from humanmock.training.train import TrainConfig, evaluate_loss, per_variable_error, predict_subject, train_member, Batcher

LITERATURE = Path(__file__).parent.parent / "data" / "observations"


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
    assert record.onset_days == 2.0 and record.path_slot == 0
    assert list(record.times) == [0.0, 2.0] and list(record.dt) == [0.0, 2.0]
    assert record.oral[0, 0] == 500.0
    assert record.obs_var[0] == OBS_INDEX["conc_slot0"]


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


def test_variable_names_cannot_reach_internal_slots():
    with pytest.raises(DataFormatError):
        variable_index("conc_slot0")
    assert variable_index("conc_paracetamol_mg_l") == OBS_INDEX["conc_slot3"]


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

    assert _readable("conc_slot0") == "conc_amoxicillin_mg_l"
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
    from humanmock.engine import Simulator

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


@pytest.mark.slow
def test_training_discovers_hidden_biology(tmp_path):
    """The real test of the whole idea.

    Data comes from a body with laws the equations do not contain. Training must
    then predict people it never saw better than the untrained body does.
    """
    generate(tmp_path / "synthetic", subjects=64, seed=3)
    records = load_records(tmp_path, include_synthetic=True)
    sample_people(records, 1, 0)
    assign_splits(records, 0)
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    config = TrainConfig(calibration_steps=30, neural_steps=20, batch_size=8, people=1, eval_every=10, patience=5)
    key = jax.random.PRNGKey(0)
    prior = BodyModel.create(jax.random.PRNGKey(0))
    before = evaluate_loss(prior, val, 8, key, TRAIN_SOLVER)
    trained, history = train_member(train, val, config, log=lambda _: None)
    after = evaluate_loss(trained, val, 8, key, TRAIN_SOLVER)
    assert np.isfinite(after)
    assert after < before - 0.05, (before, after)

    errors_before = per_variable_error(prior, val, 8, key, TRAIN_SOLVER)
    errors_after = per_variable_error(trained, val, 8, key, TRAIN_SOLVER)
    assert errors_after["throat_pain_0_10"]["rmse"] < errors_before["throat_pain_0_10"]["rmse"]

    # The trained body must still obey basic physics when served.
    from humanmock.engine import Simulator

    sim = Simulator(members=[trained], meta={"status": "trained", "model_id": "t"})
    book = DrugBook.build()
    p, _ = sim.create_particles({"age_years": 30, "weight_kg": 70}, [], book, 4, 1)
    later = sim.observe(sim.advance(p, book, 240.0, []), book)
    assert np.isfinite(later).all()
    assert later[:, OBS_INDEX["conc_slot0"]].max() == 0.0
