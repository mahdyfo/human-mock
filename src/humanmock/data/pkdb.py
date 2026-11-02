"""Downloads drug concentration curves from PK-DB and converts them for training.

PK-DB (https://pk-db.com) curates pharmacokinetics data from clinical studies.
Each study carries its own licence. Only "open" studies are fetched by default.
Studies marked "closed" are publicly viewable but have stricter terms, so you
must opt in and read data/raw/pkdb/TERMS_OF_USE.md first.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from humanmock.knowledge import KNOWLEDGE

API = "https://pk-db.com/api/v1"
HOST = "https://pk-db.com"
HEADERS = {"User-Agent": "humanmock"}
# Molecular weights convert molar units into mg per litre.
MOLAR_MASS = {"paracetamol": 151.16, "amoxicillin": 365.4, "ibuprofen": 206.28, "penicillin_v": 350.4}
SEARCH_NAMES = {"penicillin_v": "phenoxymethylpenicillin"}
SKIP_FILES = ("TabGroup", "TabIntervention", "TabIndividual", "Group", "Individual", "Intervention")
BLOOD = ("plasma", "serum", "blood")


def _number(value) -> float | None:
    """Curated tables mark gaps in several ways, so anything unreadable counts as missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _to_mg_per_l(value: float, unit: str, molar_mass: float) -> float | None:
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    table = {
        "mg/l": 1.0, "ug/ml": 1.0, "microg/ml": 1.0, "microgram/milliliter": 1.0, "milligram/liter": 1.0, "mcg/ml": 1.0,
        "ng/ml": 1e-3, "nanogram/milliliter": 1e-3, "ug/l": 1e-3, "microgram/liter": 1e-3,
        "mg/dl": 10.0, "milligram/deciliter": 10.0, "g/l": 1000.0, "gram/liter": 1000.0,
    }
    if u in table:
        return value * table[u]
    molar = {"umol/l": 1e-3, "micromole/liter": 1e-3, "nmol/l": 1e-6, "nanomole/liter": 1e-6, "nmol/ml": 1e-3,
             "mmol/l": 1.0, "millimole/liter": 1.0, "mol/l": 1e3, "um": 1e-3, "micromolar": 1e-3}
    if u in molar:
        return value * molar[u] * molar_mass
    return None


def download(raw_dir: Path, drugs: list[str], include_closed: bool = False) -> dict:
    """Saves each study's metadata and data tables under data/raw/pkdb/<drug>/."""
    summary = {}
    with httpx.Client(follow_redirects=True, timeout=240, headers=HEADERS) as client:
        for drug in drugs:
            search = SEARCH_NAMES.get(drug, drug)
            folder = raw_dir / drug
            folder.mkdir(parents=True, exist_ok=True)
            params = {"interventions__substance": search, "download": "true"}
            if not include_closed:
                params["studies__licence"] = "open"
            response = client.get(f"{API}/filter/", params=params)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                for name in ("studies.csv", "groups.csv", "individuals.csv", "interventions.csv", "TERMS_OF_USE.md", "README.md"):
                    (folder / name).write_bytes(archive.read(name))
            (raw_dir / "TERMS_OF_USE.md").write_bytes((folder / "TERMS_OF_USE.md").read_bytes())
            studies = pd.read_csv(folder / "studies.csv", dtype={"sid": str})
            files = 0
            for sid, name in zip(studies["sid"], studies["name"]):
                target = folder / "tables" / str(name)
                target.mkdir(parents=True, exist_ok=True)
                detail = client.get(f"{API}/studies/{sid}/")
                if detail.status_code != 200:
                    continue
                for entry in detail.json().get("files", []):
                    path = entry["file"]
                    base = Path(path).name
                    if not base.endswith(".tsv") or any(s in base for s in SKIP_FILES):
                        continue
                    out = target / base.lstrip(".")
                    if out.exists():
                        files += 1
                        continue
                    table = client.get(HOST + path)
                    if table.status_code == 200:
                        out.write_bytes(table.content)
                        files += 1
            summary[drug] = {"studies": int(len(studies)), "tables": files}
            print(f"  {drug}: {len(studies)} studies, {files} data tables")
    return summary


def _read_characteristics(frame: pd.DataFrame, key: str) -> dict[str, dict]:
    """Reads age, weight, height, and sex for each group or individual."""
    out: dict[str, dict] = {}
    if frame.empty or key not in frame.columns:
        return out
    for (study, owner), rows in frame.groupby(["study_name", key]):
        record: dict = {}
        for _, row in rows.iterrows():
            kind = str(row.get("measurement_type"))
            number = next((row.get(c) for c in ("value", "mean", "median") if pd.notna(row.get(c))), None)
            if kind == "age" and number is not None:
                record["age_years"] = float(number)
            elif kind == "weight" and number is not None:
                record["weight_kg"] = float(number)
            elif kind == "height" and number is not None:
                unit = str(row.get("unit"))
                record["height_cm"] = float(number) * (100.0 if unit.startswith("m") and "centi" not in unit else 1.0)
            elif kind == "sex":
                choice = str(row.get("choice"))
                if choice == "M":
                    record["sex"] = "male"
                elif choice == "F":
                    record["sex"] = "female"
        out[f"{study}|{owner}"] = record
    return out


def convert(raw_dir: Path, out_dir: Path, drugs: list[str], keep_comedication: bool = False) -> dict:
    """Writes canonical subjects, events, and observations files. Returns counts and skip reasons."""
    subjects, events, observations = [], [], []
    report: dict = {}
    for drug in drugs:
        folder = raw_dir / drug
        if not (folder / "interventions.csv").exists():
            continue
        search = SEARCH_NAMES.get(drug, drug)
        slot_name = drug
        if slot_name not in KNOWLEDGE.drug_names:
            raise ValueError(f"{drug} is not in the drug registry.")
        interventions = pd.read_csv(folder / "interventions.csv", dtype={"study_sid": str})
        interventions = interventions[interventions["normed"] == True]  # noqa: E712
        groups = _read_characteristics(pd.read_csv(folder / "groups.csv", dtype={"study_sid": str}), "group_name")
        individuals = _read_characteristics(pd.read_csv(folder / "individuals.csv", dtype={"study_sid": str}), "individual_name")
        licences = pd.read_csv(folder / "studies.csv", dtype={"sid": str}).set_index("name")["licence"].to_dict()

        skipped = {"no_dose": 0, "comedication": 0, "route": 0, "unit": 0, "repeat_dose": 0, "too_few_points": 0}
        kept = 0
        for path in sorted((folder / "tables").rglob("*.tsv")):
            try:
                table = pd.read_csv(path, sep="\t", na_values=["NA", "nan", ""], comment="#", dtype=str)
            except Exception:
                continue
            required = {"substance", "time", "intervention"}
            if not required.issubset(table.columns):
                continue
            table = table[table["substance"].str.lower() == search]
            if "tissue" in table.columns:
                table = table[table["tissue"].fillna("").str.lower().isin(BLOOD)]
            if "measurement_type" in table.columns:
                table = table[table["measurement_type"].fillna("").str.lower() == "concentration"]
            if table.empty:
                continue
            study = path.parent.name
            label_column = "label" if "label" in table.columns else None
            owner_columns = [c for c in ("group", "individual", "subject") if c in table.columns]
            keys = owner_columns + ["intervention"] + ([label_column] if label_column else [])
            for key_values, curve in table.groupby(keys, dropna=False):
                key_map = dict(zip(keys, key_values if isinstance(key_values, tuple) else (key_values,)))
                names = [n.strip() for n in str(key_map["intervention"]).split(",") if n.strip()]
                rows = interventions[(interventions["study_name"] == study) & (interventions["name"].isin(names))]
                target = rows[rows["substance_label"].str.lower() == search]
                if target.empty or target.iloc[0]["measurement_type"] != "dosing" or pd.isna(target.iloc[0]["value"]):
                    skipped["no_dose"] += 1
                    continue
                dose_row = target.iloc[0]
                # The download only lists this drug, so any other name in the list is another drug given too.
                if set(names) - set(target["name"]) and not keep_comedication:
                    skipped["comedication"] += 1
                    continue
                route = str(dose_row["route"]).lower()
                if route not in ("oral", "iv"):
                    skipped["route"] += 1
                    continue
                if str(dose_row.get("application")) not in ("single-dose", "single dose"):
                    skipped["repeat_dose"] += 1
                    continue

                owner_group = key_map.get("group")
                owner_person = key_map.get("individual") if pd.notna(key_map.get("individual")) else key_map.get("subject")
                person = individuals.get(f"{study}|{owner_person}", {}) if pd.notna(owner_person) else {}
                group = groups.get(f"{study}|{owner_group}", {}) if pd.notna(owner_group) else groups.get(f"{study}|all", {})
                traits = {**group, **person}

                unit = str(dose_row["unit"]).lower().replace(" ", "")
                amount = float(dose_row["value"])
                if unit == "gram":
                    dose_mg = amount * 1000.0
                elif unit in ("gram/kilogram", "gram/kg"):
                    dose_mg = amount * 1000.0 * traits.get("weight_kg", 70.0)
                elif unit == "milligram":
                    dose_mg = amount
                else:
                    skipped["unit"] += 1
                    continue

                points = []
                count = pd.to_numeric(curve["count"], errors="coerce").dropna() if "count" in curve else pd.Series(dtype=float)
                n = int(count.iloc[0]) if len(count) else 1
                is_arm = pd.notna(owner_group) and not pd.notna(owner_person)
                for _, row in curve.iterrows():
                    raw_value = _number(row.get("value"))
                    raw_value = raw_value if raw_value is not None else _number(row.get("mean"))
                    t = _number(row.get("time"))
                    if raw_value is None or t is None:
                        continue
                    time_unit = str(row.get("time_unit", "hr")).lower()
                    if time_unit.startswith("min"):
                        t /= 60.0
                    elif time_unit.startswith(("day", "d")):
                        t *= 24.0
                    unit_text = str(row.get("unit", ""))
                    mg_l = _to_mg_per_l(raw_value, unit_text, MOLAR_MASS[drug])
                    if mg_l is None:
                        skipped["unit"] += 1
                        continue
                    if t < 0 or t > 96 or mg_l <= 0:
                        continue
                    sd = None
                    raw_sd, raw_se = _number(row.get("sd")), _number(row.get("se"))
                    if raw_sd is not None:
                        converted = _to_mg_per_l(raw_sd, unit_text, MOLAR_MASS[drug])
                        sd = converted / max(np.sqrt(n), 1.0) if converted is not None else None
                    elif raw_se is not None:
                        sd = _to_mg_per_l(raw_se, unit_text, MOLAR_MASS[drug])
                    points.append((t, mg_l, sd))
                if len(points) < 3:
                    skipped["too_few_points"] += 1
                    continue

                subject_id = f"pkdb_{drug}_{kept:05d}"
                kept += 1
                subjects.append(
                    {
                        "subject_id": subject_id,
                        "study_id": f"pkdb:{study}",
                        "kind": "arm" if is_arm else "individual",
                        "n": n,
                        "age_years": traits.get("age_years"),
                        "sex": traits.get("sex"),
                        "weight_kg": traits.get("weight_kg"),
                        "height_cm": traits.get("height_cm"),
                        "conditions": "",
                        "split": "auto",
                        "notes": f"licence={licences.get(study, 'unknown')}; file={path.name}",
                    }
                )
                events.append(
                    {"subject_id": subject_id, "time_h": 0.0, "type": "dose", "drug": drug, "dose_mg": dose_mg, "route": route, "pathogen": ""}
                )
                for t, value, sd in points:
                    observations.append(
                        {"subject_id": subject_id, "time_h": t, "variable": f"conc_{drug}_mg_l", "value": value, "sd": sd}
                    )
        report[drug] = {"curves_kept": kept, "skipped": skipped}
        print(f"  {drug}: kept {kept} curves, skipped {skipped}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subjects).to_csv(out_dir / "subjects.csv", index=False)
    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    pd.DataFrame(observations).to_csv(out_dir / "observations.csv", index=False)
    (out_dir / "SOURCE.json").write_text(
        json.dumps({"source": "https://pk-db.com", "report": report, "restrict_redistribution": True}, indent=1), encoding="utf8"
    )
    return report


