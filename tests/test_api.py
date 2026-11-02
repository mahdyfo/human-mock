"""The session API, including memory, reset, fork, and restart."""

import pytest
from fastapi.testclient import TestClient

from humanmock.api import create_app
from humanmock.engine import Simulator

STREP_SESSION = {
    "patient": {"age_years": 30, "sex": "male", "weight_kg": 75},
    "conditions": [{"name": "strep_a", "days_since_symptom_onset": 1}],
    "people": 16,
    "seed": 11,
}
DOSE = {"type": "dose", "drug": "amoxicillin", "dose_mg": 500}


@pytest.fixture(scope="module")
def shared_sim():
    return Simulator()


@pytest.fixture()
def client(shared_sim, tmp_path):
    return TestClient(create_app(shared_sim, tmp_path / "sessions.db"))


def pain(result):
    return result["observations"]["throat_pain_0_10"]["p50"]


def test_create_reports_what_was_filled_in(client):
    r = client.post("/v1/sessions", json=STREP_SESSION)
    assert r.status_code == 201
    body = r.json()
    assert body["step"] == 0 and body["time_hours"] == 0
    assert "height_cm" in body["imputed_parameters"]
    assert "age_years" not in body["imputed_parameters"]
    assert body["model"]["status"] == "prior_only"
    assert body["support"]["within_training_support"] is False


def test_empty_request_works_with_every_default(client):
    r = client.post("/v1/sessions", json={"people": 8})
    assert r.status_code == 201
    assert "age_years" in r.json()["imputed_parameters"]


def test_session_remembers_between_calls(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    first = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [DOSE]}).json()
    assert first["step"] == 1 and first["time_hours"] == 1
    assert first["drug_levels_mg_l"]["amoxicillin"]["p50"] > 3
    # With no new dose, the drug given on the previous call must still be in the body.
    second = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h"}).json()
    assert second["step"] == 2 and second["time_hours"] == 2
    assert second["drug_levels_mg_l"]["amoxicillin"]["p50"] > 2
    current = client.get(f"/v1/sessions/{sid}").json()
    assert current["step"] == 2


def test_repeated_dosing_calls_show_slow_recovery(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    start = client.get(f"/v1/sessions/{sid}").json()
    hour = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [DOSE]}).json()
    assert abs(pain(hour) - pain(start)) < 0.4
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "7h"})
    result = None
    for _ in range(9):
        result = client.post(f"/v1/sessions/{sid}/step", json={"duration": "8h", "events": [DOSE]}).json()
    assert result["time_hours"] == 80
    assert result["share_of_people"]["culture_positive"] < 0.2
    assert pain(result) < pain(start) - 2


def test_reset_forgets_and_returns_to_identical_bodies(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    start = client.get(f"/v1/sessions/{sid}").json()
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "3d", "events": [DOSE]})
    reset = client.post(f"/v1/sessions/{sid}/reset").json()
    assert reset["step"] == 0 and reset["time_hours"] == 0
    assert reset["drug_levels_mg_l"] == {}
    assert reset["observations"] == start["observations"]
    assert len(client.get(f"/v1/sessions/{sid}/history").json()["steps"]) == 1


def test_reset_to_an_earlier_step(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    one = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1d"}).json()
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "1d"})
    back = client.post(f"/v1/sessions/{sid}/reset", json={"to_step": 1}).json()
    assert back["step"] == 1 and back["observations"] == one["observations"]
    assert client.post(f"/v1/sessions/{sid}/reset", json={"to_step": 5}).status_code == 404


def test_reset_with_new_people_draws_new_bodies(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    before = client.get(f"/v1/sessions/{sid}").json()
    after = client.post(f"/v1/sessions/{sid}/reset", json={"new_people": True, "seed": 99}).json()
    assert after["seed"] == 99
    assert after["observations"] != before["observations"]


def test_fork_tries_two_treatments_on_identical_bodies(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h"})
    fork = client.post(f"/v1/sessions/{sid}/fork").json()
    assert fork["forked_from"]["session_id"] == sid and fork["step"] == 1
    fid = fork["session_id"]
    treated = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1d", "events": [DOSE]}).json()
    untreated = client.post(f"/v1/sessions/{fid}/step", json={"duration": "1d"}).json()
    assert treated["observations"]["throat_culture_log10_cfu"]["p50"] < untreated["observations"]["throat_culture_log10_cfu"]["p50"]


def test_sessions_survive_a_server_restart(shared_sim, tmp_path):
    database = tmp_path / "sessions.db"
    first = TestClient(create_app(shared_sim, database))
    sid = first.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    before = first.post(f"/v1/sessions/{sid}/step", json={"duration": "8h", "events": [DOSE]}).json()
    restarted = TestClient(create_app(Simulator(), database))
    current = restarted.get(f"/v1/sessions/{sid}").json()
    assert current["step"] == before["step"] and current["observations"] == before["observations"]
    assert restarted.post(f"/v1/sessions/{sid}/step", json={"duration": "1h"}).json()["step"] == 2


def test_a_dosing_schedule_in_one_call_matches_separate_calls(client):
    a = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    b = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    together = client.post(
        f"/v1/sessions/{a}/step", json={"duration": "1d", "events": [{**DOSE, "repeat": 3, "every_hours": 8}]}
    ).json()
    separate = None
    for _ in range(3):
        separate = client.post(f"/v1/sessions/{b}/step", json={"duration": "8h", "events": [DOSE]}).json()
    assert together["time_hours"] == separate["time_hours"] == 24
    x, y = together["observations"]["throat_pain_0_10"]["p50"], separate["observations"]["throat_pain_0_10"]["p50"]
    assert abs(x - y) < 0.1


def test_bad_requests_explain_themselves(client):
    assert client.post("/v1/sessions", json={"patient": {"weigth_kg": 70}}).status_code == 422
    assert client.post("/v1/sessions", json={"patient": {"age_years": 5}}).status_code == 422
    assert client.post("/v1/sessions", json={"conditions": [{"name": "plague"}]}).status_code == 422
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    unknown = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{**DOSE, "drug": "magic"}]})
    assert unknown.status_code == 422 and "amoxicillin" in unknown.json()["detail"]
    outside = client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{**DOSE, "at_hours": 3}]})
    assert outside.status_code == 422
    assert client.post(f"/v1/sessions/{sid}/step", json={"duration": "soon"}).status_code == 422
    assert client.get("/v1/sessions/nope").status_code == 404


def test_a_failed_step_leaves_the_session_untouched(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{**DOSE, "drug": "magic"}]})
    assert client.get(f"/v1/sessions/{sid}").json()["step"] == 0


def test_warnings_for_dose_limits_and_allergies(client):
    body = {**STREP_SESSION, "patient": {**STREP_SESSION["patient"], "allergies": ["amoxicillin"]}}
    sid = client.post("/v1/sessions", json=body).json()["session_id"]
    result = client.post(
        f"/v1/sessions/{sid}/step",
        json={"duration": "1d", "events": [{"type": "dose", "drug": "amoxicillin", "dose_mg": 2000, "repeat": 2, "every_hours": 8}]},
    ).json()
    text = " ".join(result["warnings"])
    assert "allergy" in text and "single-dose" in text and "24 h" in text


def test_a_daily_limit_counts_doses_from_earlier_calls(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    for _ in range(3):
        result = client.post(
            f"/v1/sessions/{sid}/step",
            json={"duration": "2h", "events": [{"type": "dose", "drug": "paracetamol", "dose_mg": 1000}]},
        ).json()
        assert result["warnings"] == []
    extra = client.post(
        f"/v1/sessions/{sid}/step",
        json={"duration": "2h", "events": [{"type": "dose", "drug": "paracetamol", "dose_mg": 1000, "repeat": 2, "every_hours": 0.5}]},
    ).json()
    assert any("24 h" in w for w in extra["warnings"])


def test_hypothetical_drug_through_the_api(client):
    body = {
        **STREP_SESSION,
        "custom_drugs": [{"name": "candidate_x", "kill_emax": 2.0, "kill_ec50": 0.05, "reservoir_penetration": 0.5}],
    }
    sid = client.post("/v1/sessions", json=body).json()["session_id"]
    result = client.post(
        f"/v1/sessions/{sid}/step",
        json={"duration": "1d", "events": [{"type": "dose", "drug": "candidate_x", "dose_mg": 500}]},
    ).json()
    assert any("hypothetical" in w for w in result["warnings"])
    assert result["drug_levels_mg_l"]["candidate_x"]["p50"] >= 0


def test_internal_state_is_available_on_request(client):
    body = {**STREP_SESSION, "include_internal": True}
    result = client.post("/v1/sessions", json=body).json()
    assert "log_surface" in result["internal"] and "memory_neuron_activity" in result["internal"]
    assert "internal" not in client.post("/v1/sessions", json=STREP_SESSION).json()


def test_catalog_and_model_describe_the_service(client):
    catalog = client.get("/v1/catalog").json()
    assert "amoxicillin" in catalog["drugs"] and "strep_a" in catalog["conditions"]
    model = client.get("/v1/model").json()
    assert model["status"] == "prior_only" and model["trainable_parameters"]["organ_cells"] > 0


def test_the_web_page_is_served_without_hiding_the_api(client):
    page = client.get("/")
    assert page.status_code == 200 and "humanmock" in page.text
    for name in ("app.js", "body.js", "chart.js", "api.js", "style.css", "fonts/atkinson-hyperlegible-next.woff2"):
        assert client.get(f"/{name}").status_code == 200, name
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200


def test_the_page_only_loads_files_that_exist_locally(client):
    """The page must work offline, so it never fetches scripts or styles from elsewhere."""
    page = client.get("/").text
    assert "<script" in page and 'src="http' not in page and 'href="http' not in page.split("<body>")[0]
    style = client.get("/style.css").text
    assert "http" not in style


def test_the_catalog_describes_the_body_so_a_page_can_draw_it(client):
    catalog = client.get("/v1/catalog").json()
    organs = {o["name"]: o for o in catalog["organs"]}
    pain = next(m for m in organs["throat"]["measurements"] if m["name"] == "throat_pain_0_10")
    assert pain["label"] == "Throat pain" and pain["body_part"] == "throat" and pain["severe"] > pain["healthy"]
    assert not any(m["name"].startswith("conc_slot") for o in catalog["organs"] for m in o["measurements"])
    assert any(s["key"] == "fever" and s["measurement"] == "temperature_c" for s in organs["thermoregulation"]["shares"])
    assert "cytokine" in [s["name"] for s in organs["inflammation"]["states"]]
    assert catalog["patient"]["weight_kg"]["note"]


def test_the_session_list_says_who_each_session_is_about(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    client.post(f"/v1/sessions/{sid}/step", json={"duration": "8h"})
    fork = client.post(f"/v1/sessions/{sid}/fork").json()["session_id"]
    listed = {s["session_id"]: s for s in client.get("/v1/sessions").json()["sessions"]}
    assert listed[sid]["patient"]["age_years"] == 30 and listed[sid]["time_hours"] == 8
    assert listed[sid]["conditions"][0]["name"] == "strep_a" and listed[sid]["forked_from"] is None
    assert listed[fork]["forked_from"] == {"session_id": sid, "step": 1}
    assert client.get(f"/v1/sessions/{fork}").json()["forked_from"]["session_id"] == sid


def test_an_overdose_never_looks_like_proof_of_safety(client):
    """This simulator has no severe poisoning or death, so recovering must not read as safe."""
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    normal = client.post(
        f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{"drug": "paracetamol", "dose_mg": 500}]}
    ).json()
    assert normal["warnings"] == []
    overdose = client.post(
        f"/v1/sessions/{sid}/step", json={"duration": "1d", "events": [{"drug": "paracetamol", "dose_mg": 15000}]}
    ).json()
    text = " ".join(overdose["warnings"])
    assert "cannot show severe poisoning or death" in text and "poison centre" in text


def test_one_dose_of_several_kilograms_is_refused(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    too_much = client.post(
        f"/v1/sessions/{sid}/step", json={"duration": "1h", "events": [{"drug": "amoxicillin", "dose_mg": 5000000}]}
    )
    assert too_much.status_code == 422


def test_a_session_from_another_model_explains_how_to_continue(shared_sim, tmp_path):
    from humanmock.engine import Simulator as Sim

    database = tmp_path / "sessions.db"
    sid = TestClient(create_app(shared_sim, database)).post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    retrained = Sim(members=shared_sim.members, meta={"status": "trained", "model_id": "another", "support": {}})
    client = TestClient(create_app(retrained, database))
    refused = client.get(f"/v1/sessions/{sid}")
    assert refused.status_code == 409 and "new_people" in refused.json()["detail"]
    # What the page needs in order to offer the way out.
    assert client.get(f"/v1/sessions/{sid}/history").status_code == 200
    assert client.post(f"/v1/sessions/{sid}/reset", json={"new_people": True}).status_code == 200
    assert client.get(f"/v1/sessions/{sid}").status_code == 200


def test_delete_removes_the_session(client):
    sid = client.post("/v1/sessions", json=STREP_SESSION).json()["session_id"]
    assert client.delete(f"/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/v1/sessions/{sid}").status_code == 404
    assert client.delete(f"/v1/sessions/{sid}").status_code == 404
