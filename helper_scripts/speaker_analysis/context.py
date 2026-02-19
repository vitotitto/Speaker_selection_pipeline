from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PatientContext:
    """Enriched metadata about the patient for the LLM prompt."""
    name: str
    group: str
    dementia_type: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    first_symptoms_year: Optional[int] = None
    timepoint: str = ""
    video_stem: str = ""
    language: Optional[str] = None


def _safe_int(val: str) -> Optional[int]:
    if not val or not val.strip():
        return None
    try:
        return int(float(val.strip()))
    except (ValueError, TypeError):
        return None


def _load_csv_by_name(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a CSV and return rows keyed by the name column (case-insensitive)."""
    result: Dict[str, Dict[str, Any]] = {}
    if not csv_path.exists():
        return result
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        # Skip leading blank lines (some CSVs have empty rows before headers)
        lines = f.readlines()
        cleaned = [line for line in lines if line.strip()]
        if not cleaned:
            return result
        reader = csv.DictReader(cleaned)
        for row in reader:
            name = (row.get("name") or row.get("Name") or "").strip()
            if name:
                result[name.lower()] = row
    return result


_dementia_cache: Optional[Dict[str, Dict[str, Any]]] = None
_no_dementia_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _get_dementia_lookup(csv_dir: Path) -> Dict[str, Dict[str, Any]]:
    global _dementia_cache
    if _dementia_cache is None:
        _dementia_cache = _load_csv_by_name(
            csv_dir / "DementiaNet_merged_completed_patients_with_dementia.csv"
        )
    return _dementia_cache


def _get_no_dementia_lookup(csv_dir: Path) -> Dict[str, Dict[str, Any]]:
    global _no_dementia_cache
    if _no_dementia_cache is None:
        _no_dementia_cache = _load_csv_by_name(
            csv_dir / "DementiaNet - no dementia.csv"
        )
    return _no_dementia_cache


def build_patient_context(
    person_name: str,
    source: str,
    timepoint: str,
    video_stem: str,
    csv_dir: Path,
) -> PatientContext:
    """Look up the person in the correct CSV and build enriched context."""
    is_dementia = "dementia" in source.lower() and "no_dementia" not in source.lower()
    group = "Dementia" if is_dementia else "No Dementia"
    key = person_name.strip().lower()

    ctx = PatientContext(
        name=person_name,
        group=group,
        timepoint=timepoint,
        video_stem=video_stem,
    )

    if is_dementia:
        lookup = _get_dementia_lookup(csv_dir)
        row = lookup.get(key)
        if row:
            ctx.dementia_type = (row.get("dementia type") or "").strip() or None
            ctx.gender = (row.get("gender") or "").strip() or None
            ctx.ethnicity = (row.get("ethnicity") or "").strip() or None
            ctx.birth_year = _safe_int(row.get("birth", ""))
            ctx.death_year = _safe_int(row.get("death", ""))
            ctx.first_symptoms_year = _safe_int(row.get("first symptoms", ""))
            ctx.language = (row.get("language") or "").strip() or None
    else:
        lookup = _get_no_dementia_lookup(csv_dir)
        row = lookup.get(key)
        if row:
            ctx.gender = (row.get("gender") or "").strip() or None
            ctx.ethnicity = (row.get("ethnicity") or "").strip() or None
            ctx.birth_year = _safe_int(row.get("birthdate", ""))
            ctx.death_year = _safe_int(row.get("deathdate", ""))
            ctx.language = (row.get("language") or "").strip() or None

    return ctx
