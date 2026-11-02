"""Finds trials with posted results on ClinicalTrials.gov, ready for you to curate.

Trial results are reported in many different shapes, so they cannot be converted
safely by a script. This writes a review sheet of candidate outcomes. A person
then decides which ones truly match a model variable, and copies those into
data/observations/custom/ with a note saying where each number came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

API = "https://clinicaltrials.gov/api/v2/studies"


def download(raw_dir: Path, condition: str = "streptococcal pharyngitis", limit: int = 100) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "query.cond": condition,
        "filter.overallStatus": "COMPLETED",
        "aggFilters": "results:with",
        "pageSize": min(limit, 100),
        "format": "json",
    }
    studies, token = [], None
    with httpx.Client(timeout=120, headers={"User-Agent": "humanmock"}) as client:
        while len(studies) < limit:
            if token:
                params["pageToken"] = token
            response = client.get(API, params=params)
            response.raise_for_status()
            body = response.json()
            studies.extend(body.get("studies", []))
            token = body.get("nextPageToken")
            if not token:
                break
    slug = condition.replace(" ", "_")
    (raw_dir / f"{slug}.json").write_text(json.dumps(studies), encoding="utf8")

    rows = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        nct = protocol.get("identificationModule", {}).get("nctId")
        title = protocol.get("identificationModule", {}).get("briefTitle")
        measures = study.get("resultsSection", {}).get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
        for measure in measures:
            rows.append(
                {
                    "nct_id": nct,
                    "study": title,
                    "outcome": measure.get("title"),
                    "timeframe": measure.get("timeFrame"),
                    "type": measure.get("paramType"),
                    "unit": measure.get("unitOfMeasure"),
                    "groups": len(measure.get("groups", [])),
                    "link": f"https://clinicaltrials.gov/study/{nct}?tab=results",
                }
            )
    sheet = raw_dir / f"{slug}_review_sheet.csv"
    pd.DataFrame(rows).to_csv(sheet, index=False)
    return {"studies": len(studies), "outcomes": len(rows), "review_sheet": str(sheet)}
