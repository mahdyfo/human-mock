"""Rules every organ and every constant must follow.

These tests read the organ list, so a new organ is covered without writing a test.
A failure here means a contribution does not fit the body yet.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import humanmock.model as model_module
from humanmock.data.formats import read_source, sample_people
from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE, MAX_DRUGS, RAW, SPARE_DRUG_SLOTS, KnowledgeError, evidence_gaps, load_knowledge, validate_knowledge
from humanmock.model import COV_CENTER, LAYOUT, OBS_NAMES, PUBLIC_OBSERVABLES, TRAIN_SOLVER, BodyModel, OrganError, _publish_signals, check_organs
from humanmock.organs import ORGANS
from humanmock.organs.base import FRACTION, Measurement, Organ, Share, StateDef
from humanmock.training.train import Batcher, TrainConfig, _loss_batch

BY_NAME = {o.name: o for o in ORGANS}


def typical_context(model):
    return model.context(
        jnp.asarray(COV_CENTER),
        jnp.zeros(len(KNOWLEDGE.sys_names)),
        jnp.zeros((KNOWLEDGE.n_pathogens, len(KNOWLEDGE.path_fields))),
        jnp.zeros((MAX_DRUGS, len(DRUG_FIELDS))),
        jnp.asarray(KNOWLEDGE.drug_table),
    )


def sick_state(model, c):
    """A person with an established infection and a drug on board, to reach the busier parts of the equations."""
    y = model.initial_state(c)
    return model.apply_events(y, c, jnp.full(MAX_DRUGS, 100.0), jnp.full(MAX_DRUGS, 100.0), jnp.full(KNOWLEDGE.n_pathogens, 6.0))


@pytest.fixture(scope="module")
def body():
    model = BodyModel.create(jax.random.PRNGKey(0))
    c = typical_context(model)
    return model, c, {"healthy": model.initial_state(c), "sick": sick_state(model, c)}


@pytest.mark.parametrize("organ", ORGANS, ids=lambda o: o.name)
@pytest.mark.parametrize("condition", ["healthy", "sick"])
def test_an_organ_gives_a_finite_rate_for_exactly_its_own_states(organ, condition, body):
    _, c, states = body
    s = LAYOUT.unpack(states[condition])
    rates = organ.derivative(s, c, _publish_signals(s, c))
    assert set(rates) == {st.name for st in organ.states}
    for st in organ.states:
        assert jnp.shape(rates[st.name]) in ((), (st.size,)), f"{st.name} has the wrong shape."
        assert np.isfinite(np.asarray(rates[st.name])).all(), f"{st.name} is not finite when {condition}."


@pytest.mark.parametrize("organ", [o for o in ORGANS if o.measurements], ids=lambda o: o.name)
@pytest.mark.parametrize("condition", ["healthy", "sick"])
def test_an_organ_gives_a_finite_value_for_exactly_its_measurements(organ, condition, body):
    _, c, states = body
    s = LAYOUT.unpack(states[condition])
    values = organ.observables(s, c, _publish_signals(s, c))
    assert set(values) == {m.name for m in organ.measurements}
    for m in organ.measurements:
        assert jnp.shape(values[m.name]) == (), f"{m.name} must be a single number."
        assert np.isfinite(np.asarray(values[m.name])), f"{m.name} is not finite when {condition}."


@pytest.mark.parametrize("organ", ORGANS, ids=lambda o: o.name)
def test_an_organ_is_documented(organ):
    assert organ.description, f"Organ {organ.name} needs a description."
    for st in organ.states:
        assert st.rate_scale > 0, f"State {st.name} needs a rate_scale above 0, or its neural cell cannot learn."
    for m in organ.measurements:
        assert m.unit, f"Measurement {m.name} needs a unit."
        assert m.noise > 0, f"Measurement {m.name} needs noise above 0."


def test_an_untrained_body_is_exactly_the_equations(body):
    model, c, states = body
    y = states["sick"]
    s = LAYOUT.unpack(y)
    e = _publish_signals(s, c)
    expected = {}
    for organ in ORGANS:
        expected.update(organ.derivative(s, c, e))
    rates = model.vector_field(0.0, y, c)
    for name, off, size, _, _ in LAYOUT.entries:
        assert np.allclose(np.asarray(rates[off : off + size]), np.asarray(expected[name])), name
    assert np.allclose(np.asarray(rates[LAYOUT.physio_size :]), 0.0)


def test_every_measurement_can_be_produced(body):
    model, c, states = body
    assert model.observe(states["sick"], c).shape == (len(OBS_NAMES),)
    assert all(unit for unit in PUBLIC_OBSERVABLES.values())


def test_values_cannot_be_mislabelled_by_the_order_an_organ_returns_them(body, monkeypatch):
    """An organ may return its values in any order, because they are read by name."""
    model, c, states = body
    expected = np.asarray(model.observe(states["sick"], c))
    organ = BY_NAME["thermoregulation"]

    def reversed_order(s, c, e):
        return dict(reversed(list(organ.observables(s, c, e).items())))

    shuffled = dataclasses.replace(organ, observables=reversed_order)
    monkeypatch.setattr(model_module, "ORGANS", tuple(shuffled if o is organ else o for o in ORGANS))
    assert np.array_equal(np.asarray(model.observe(states["sick"], c)), expected)


def test_gradients_through_a_long_illness_stay_a_sensible_size(tmp_path):
    """Training only works if gradients stay meaningful, so this covers every organ.

    If this fails after adding an organ, one of its states probably changes faster than
    the training solver can follow, at more than about 2.5 per hour. Slow that state
    down, or lower dtmax in TRAIN_SOLVER, which makes training slower.
    """
    pd.DataFrame(
        [{"subject_id": "a", "study_id": "s", "age_years": 40, "weight_kg": 70, "conditions": f"{KNOWLEDGE.path_names[0]}:1"}]
    ).to_csv(tmp_path / "subjects.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "a", "time_h": 8.0 * k, "type": "dose", "drug": KNOWLEDGE.drug_names[0], "dose_mg": 500, "route": "oral"}
            for k in range(9)
        ]
    ).to_csv(tmp_path / "events.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "a", "time_h": t, "variable": name, "value": 1.0}
            for t in (24, 96, 240)
            for name in PUBLIC_OBSERVABLES
        ]
    ).to_csv(tmp_path / "observations.csv", index=False)
    records = read_source(tmp_path)
    sample_people(records, 4, 0)
    batch = Batcher(records, 1, 0).collate(records, 32, 64)
    model = BodyModel.create(jax.random.PRNGKey(0))

    def loss(m):
        return _loss_batch(m, batch, jax.random.PRNGKey(1), TrainConfig(), TRAIN_SOLVER)[0]

    gradients = eqx.filter_jit(eqx.filter_grad(loss))(model)
    leaves = [np.asarray(g) for g in jax.tree_util.tree_leaves(eqx.filter(gradients, eqx.is_inexact_array))]
    assert all(np.isfinite(leaf).all() for leaf in leaves), "A gradient is not finite."
    size = float(np.sqrt(sum((leaf**2).sum() for leaf in leaves)))
    # These targets are far from any real value, which alone gives a size of about 1000.
    # An unstable state gives hundreds of millions, so this limit separates the two clearly.
    assert size < 1e5, f"The gradient size is {size:.3g}, which would stop training from working."


def _organ(**changes) -> Organ:
    settings = dict(
        name="new",
        description="d",
        states=(StateDef("new_state", FRACTION, rate_scale=0.01),),
        neighbors=(),
        derivative=lambda s, c, e: {"new_state": 0.0},
    )
    return Organ(**{**settings, **changes})


@pytest.mark.parametrize(
    "organ,message",
    [
        (_organ(neighbors=("liverr",)), "unknown neighbor liverr"),
        (_organ(states=(StateDef("temperature", FRACTION, rate_scale=0.01),)), "state named temperature"),
        (_organ(states=(StateDef("x", "fractionn", rate_scale=0.01),)), "unknown kind"),
        (_organ(states=(StateDef("x", FRACTION, rate_scale=0.01, per="organ"),)), "unknown per"),
        (_organ(states=(StateDef("x", FRACTION, rate_scale=0.01, init_from="shoe_size"),)), "shoe_size"),
        (_organ(measurements=(Measurement("x_mg", "mg"),)), "no observables"),
        (_organ(shares=(Share("x_fraction", "nothing_real", 1.0, 0.1),)), "nothing_real"),
    ],
)
def test_a_badly_declared_organ_gets_a_clear_message(organ, message):
    problems = check_organs(list(ORGANS) + [organ])
    assert any(message in p for p in problems), problems
    assert issubclass(OrganError, ValueError)


def test_a_misspelled_drug_field_is_refused_with_a_suggestion():
    raw = {**RAW, "drugs": {**RAW["drugs"], "typo": {"label": "x", "sources": {}, "kil_emax": 1.6}}}
    problems = validate_knowledge(raw)
    assert any("kil_emax" in p and "kill_emax" in p for p in problems), problems
    with pytest.raises(KnowledgeError, match="kil_emax"):
        load_knowledge(raw)


@pytest.mark.parametrize(
    "section,change,message",
    [
        ("system", {"k_new": {"value": 1.0, "iiv": 0.1, "note": "n"}}, "system.k_new needs a source"),
        ("system", {"k_new": {"value": 1.0, "source": "s", "note": "n"}}, "system.k_new needs a iiv"),
        ("patient", {"new": {"center": 1, "scale": 1, "note": "n", "prior": ["normal", 1, 1]}}, "needs a source"),
        ("patient", {"new": {"center": 1, "scale": 1, "note": "n", "prior": ["uniform", 1, 1], "source": "s"}}, "prior must be"),
        ("drugs", {"new": {"label": "x", "ka": 1.0}}, "drugs.new needs sources"),
    ],
)
def test_a_constant_without_a_source_is_refused(section, change, message):
    problems = validate_knowledge({**RAW, section: {**RAW[section], **change}})
    assert any(message in p for p in problems), problems


def test_a_new_pathogen_must_have_every_field():
    partial = {"label": "x", "r": {"value": 1.0, "iiv": 0.1, "source": "s", "note": "n"}}
    problems = validate_knowledge({**RAW, "pathogens": {**RAW["pathogens"], "new": partial}})
    assert any("pathogens.new is missing" in p for p in problems), problems


def test_there_is_always_room_for_hypothetical_drugs():
    assert MAX_DRUGS == len(KNOWLEDGE.drug_names) + SPARE_DRUG_SLOTS


def test_the_list_of_missing_evidence_only_names_uncited_constants():
    gaps, total = evidence_gaps(RAW)
    named = dict(gaps)
    assert 0 < len(gaps) <= total
    assert "system.k_heal" in named
    assert "drugs.amoxicillin.pk" not in named
