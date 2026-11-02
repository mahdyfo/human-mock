"""Follows the steps in CONTRIBUTING.md on a copy of the project.

If these steps ever stop working, this test fails, so the guide cannot quietly go out
of date. It adds the example organ, a drug, and a patient parameter, then checks they
reach the API, the data format, and the neural network without any other edits.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "add_an_organ"


def follow_the_guide(project: Path) -> None:
    shutil.copytree(ROOT / "src" / "humanmock", project / "humanmock", ignore=shutil.ignore_patterns("__pycache__"))

    # Step 1: put the organ file in the organs folder.
    shutil.copy(EXAMPLE / "kidney.py", project / "humanmock" / "organs" / "kidney.py")

    # Step 2: add one import and one list entry.
    listing = project / "humanmock" / "organs" / "__init__.py"
    text = listing.read_text(encoding="utf8")
    text = text.replace("from . import immunity,", "from . import kidney, immunity,")
    text = text.replace("    systemic.ORGAN,\n", "    systemic.ORGAN,\n    kidney.ORGAN,\n")
    listing.write_text(text, encoding="utf8")

    # Step 3: add its constants, with sources, to the knowledge base.
    knowledge = project / "humanmock" / "knowledge_data" / "physiology.yaml"
    raw = yaml.safe_load(knowledge.read_text(encoding="utf8"))
    for section, entries in yaml.safe_load((EXAMPLE / "additions.yaml").read_text(encoding="utf8")).items():
        raw[section].update(entries)
    knowledge.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf8")


CHECK = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    import humanmock
    from humanmock.api import create_app
    from humanmock.data.formats import check_source
    from humanmock.engine import Simulator
    from humanmock.knowledge import MAX_DRUGS, SPARE_DRUG_SLOTS, KNOWLEDGE
    from humanmock.model import BodyModel, COVARIATES
    from humanmock.organs import ORGANS

    assert Path(sys.argv[1]) in Path(humanmock.__file__).parents, humanmock.__file__
    assert ORGANS[-1].name == "kidney"

    sim = Simulator()
    assert len(sim.members[0].cells) == len(ORGANS), "The new organ did not get a neural cell."
    assert "urine_output_l_day" in COVARIATES
    assert MAX_DRUGS == len(KNOWLEDGE.drug_names) + SPARE_DRUG_SLOTS

    client = TestClient(create_app(sim, Path(sys.argv[1]) / "s.db"))
    catalog = client.get("/v1/catalog").json()
    assert "creatinine_rise_mg_dl" in catalog["observables"]
    assert catalog["observables"]["creatinine_rise_mg_dl"] == "mg per decilitre"
    assert "example_drug" in catalog["drugs"]
    assert "urine_output_l_day" in catalog["patient_parameters"]

    created = client.post(
        "/v1/sessions",
        json={"patient": {"age_years": 40, "urine_output_l_day": 1.2}, "people": 4, "seed": 1, "include_internal": True},
    )
    assert created.status_code == 201, created.text
    start = created.json()
    sid = start["session_id"]
    step = client.post(
        f"/v1/sessions/{sid}/step",
        json={"duration": "24h", "events": [{"drug": "example_drug", "dose_mg": 2000}], "include_internal": True},
    )
    assert step.status_code == 200, step.text
    body = step.json()
    assert body["observations"]["creatinine_rise_mg_dl"]["p50"] > start["observations"]["creatinine_rise_mg_dl"]["p50"]
    assert "kidney_injury" in body["internal"]
    assert "kidney_injury" in body["share_of_people"]
    assert "example_drug" in body["drug_levels_mg_l"]

    report = check_source(Path(sys.argv[2]))
    assert not report.errors, report.errors
    assert report.subjects == 1 and report.measurements == 2

    print("the guide works")
    """
)


@pytest.mark.parametrize("target", ["api and data", "the rules for organs"])
def test_following_the_guide_works(tmp_path, target):
    project = tmp_path / "project"
    follow_the_guide(project)
    env = {**os.environ, "PYTHONPATH": str(project), "PYTHONUTF8": "1"}

    if target == "the rules for organs":
        # The rules read the organ list, so the new organ is checked with no new test.
        done = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_contract.py"), "-q", "-x", "-p", "no:cacheprovider",
             "--rootdir", str(tmp_path), "-c", os.devnull],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=600,
        )
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
        assert "kidney" in subprocess.run(
            [sys.executable, "-c", "from humanmock.organs import ORGANS; print([o.name for o in ORGANS])"],
            env=env, cwd=tmp_path, capture_output=True, text=True,
        ).stdout
        return

    source = tmp_path / "smith_2020"
    source.mkdir()
    pd.DataFrame(
        [{"subject_id": "a", "study_id": "smith_2020", "kind": "individual", "age_years": 40, "urine_output_l_day": 1.2, "notes": "example"}]
    ).to_csv(source / "subjects.csv", index=False)
    pd.DataFrame(
        [{"subject_id": "a", "time_h": 0, "type": "dose", "drug": "example_drug", "dose_mg": 500, "route": "oral", "pathogen": ""}]
    ).to_csv(source / "events.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "a", "time_h": 2, "variable": "conc_example_drug_mg_l", "value": 5.0, "sd": ""},
            {"subject_id": "a", "time_h": 24, "variable": "creatinine_rise_mg_dl", "value": 0.2, "sd": ""},
        ]
    ).to_csv(source / "observations.csv", index=False)
    script = tmp_path / "check.py"
    script.write_text(CHECK, encoding="utf8")
    done = subprocess.run(
        [sys.executable, str(script), str(project), str(source)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "the guide works" in done.stdout
