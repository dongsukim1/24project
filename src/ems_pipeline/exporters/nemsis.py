"""NEMSIS v3.5 data-element exporter.

Maps a ``CanonicalClaim`` to a Python dict whose keys follow the NEMSIS v3.5
element naming convention (e.g. ``ePatient.13`` for sex).

This is an MVP mapping layer.  It is *not* an XML serialiser; the dict it
produces can be used as a structured intermediate for downstream NEMSIS XML
generation.

NEMSIS reference: https://nemsis.org/technical-resources/version-3/

Key sections covered:
    eResponse   — unit, run number, agency
    eTimes      — incident timeline
    ePatient    — patient demographics
    eSituation  — chief complaint, primary impression
    eVitals     — vital signs
    eProcedures — procedures
    eMedications— medications administered
    eDisposition— transport disposition
    eBilling    — billing / payer info (eCustomResults extension)
"""

from __future__ import annotations

from typing import Any

from ems_pipeline.claim.canonical import CanonicalClaim
from ems_pipeline.exporters import ExportResult

# ---------------------------------------------------------------------------
# Required NEMSIS fields (simplified MVP list)
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "eResponse.AgencyIncidentCallTypeGroup.eResponse.05",  # type of service requested
    "eTimes.01",   # PSAP call date/time
    "eTimes.05",   # unit on-scene date/time
    "eTimes.12",   # patient contact
    "ePatient.13", # sex
    "eSituation.11",  # provider primary impression
    "eDisposition.27",  # unit disposition
]

_OPTIONAL_FIELDS = [
    "ePatient.02",   # patient last name
    "ePatient.14",   # race
    "ePatient.16",   # age
    "eTimes.06",     # patient contact datetime
    "eTimes.09",     # transport date/time
    "eTimes.11",     # arrive destination
    "eVitals",       # at least one vital
    "eProcedures",   # at least one procedure
    "eMedications",  # at least one medication
]


# ---------------------------------------------------------------------------
# NEMSIS value maps
# ---------------------------------------------------------------------------

_SEX_MAP = {"M": "9906001", "F": "9906003", "U": "9906009"}

_DISPOSITION_MAP = {
    "transported": "4227001",  # Patient Transported
    "refused":     "4227007",  # Treated, Released (Patient Refused)
    "canceled":    "4227013",  # Cancelled (Prior to Arrival)
    "unknown":     "4227023",  # No Treatment / Transport Required (Investigation)
}


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _fmt_dt(dt: Any) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _export_eresponse(claim: CanonicalClaim) -> dict[str, Any]:
    return {
        "eResponse.01": claim.encounter.agency_id,
        "eResponse.02": claim.encounter.agency_name,
        "eResponse.03": claim.encounter.run_number,
        "eResponse.05": "2205001",  # 911 Response (default; no source data)
        "eResponse.07": claim.unit.unit_id,
        "eResponse.13": claim.encounter.cad_incident_number,
        "eResponse.14": claim.encounter.pcr_number,
    }


def _export_etimes(claim: CanonicalClaim) -> dict[str, Any]:
    t = claim.timeline
    return {
        "eTimes.01": _fmt_dt(t.psap_call_datetime),
        "eTimes.03": _fmt_dt(t.dispatch_notified_datetime),
        "eTimes.05": _fmt_dt(t.unit_enroute_datetime),
        "eTimes.06": _fmt_dt(t.unit_on_scene_datetime),
        "eTimes.07": _fmt_dt(t.patient_contact_datetime),
        "eTimes.09": _fmt_dt(t.transport_begin_datetime),
        "eTimes.11": _fmt_dt(t.arrive_destination_datetime),
        "eTimes.12": _fmt_dt(t.transfer_of_care_datetime),
    }


def _export_epatient(claim: CanonicalClaim) -> dict[str, Any]:
    p = claim.patient
    name_parts = (p.full_name or "").split(" ", 1)
    first = name_parts[0] if name_parts else None
    last = name_parts[1] if len(name_parts) > 1 else None
    race_codes = [rv.code for rv in p.race] if p.race else []
    return {
        "ePatient.02": last,
        "ePatient.03": first,
        "ePatient.13": _SEX_MAP.get(p.sex_at_birth or "", "9906009"),
        "ePatient.14": race_codes or None,
        "ePatient.15": None,  # ethnicity (stub)
        "ePatient.16": p.age_years,
        "ePatient.17": "2516009" if p.age_years is not None else None,  # age units: Years
        "ePatient.18": (p.address.city if p.address else None),
        "ePatient.19": (p.address.state if p.address else None),
        "ePatient.20": (p.address.zip_code if p.address else None),
    }


def _export_esituation(claim: CanonicalClaim) -> dict[str, Any]:
    c = claim.clinical
    impression_code = c.primary_impression.code if c.primary_impression else None
    impression_display = c.primary_impression.display if c.primary_impression else None
    return {
        "eSituation.04": c.chief_complaint,
        "eSituation.11": impression_code,
        "eSituation.11_display": impression_display,
        "eSituation.09": list(c.symptoms) if c.symptoms else None,
    }


def _export_evitals(claim: CanonicalClaim) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in claim.clinical.vitals:
        out.append({
            "eVitals.01": _fmt_dt(v.timestamp),
            "eVitals.name": v.name,
            "eVitals.value": v.value,
            "eVitals.unit": v.unit,
            "eVitals.method": v.method,
            "eVitals.evidence_segment_ids": v.evidence_segment_ids,
        })
    return out


def _export_eprocedures(claim: CanonicalClaim) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in claim.clinical.procedures:
        out.append({
            "eProcedures.01": _fmt_dt(p.timestamp),
            "eProcedures.03": p.coded.code if p.coded else None,
            "eProcedures.03_system": p.coded.code_system if p.coded else None,
            "eProcedures.03_display": p.name,
            "eProcedures.05": p.performed_by,
        })
    return out


def _export_emedications(claim: CanonicalClaim) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in claim.clinical.medications:
        out.append({
            "eMedications.01": _fmt_dt(m.timestamp),
            "eMedications.03": m.coded.code if m.coded else None,
            "eMedications.03_display": m.drug,
            "eMedications.05": m.dose,
            "eMedications.06": m.route,
            "eMedications.07": m.administered_by,
        })
    return out


def _export_edisposition(claim: CanonicalClaim) -> dict[str, Any]:
    t = claim.transport
    disp_code = _DISPOSITION_MAP.get(t.disposition_status or "unknown", "4227023")
    return {
        "eDisposition.02": (
            t.destination_address.street if t.destination_address else None
        ),
        "eDisposition.03": t.destination_facility_name,
        "eDisposition.06": t.destination_facility_id,
        "eDisposition.17": "4217003",  # Destination type: Hospital (default stub)
        "eDisposition.27": disp_code,
        "eDisposition.28": t.disposition_status,
    }


def _export_ebilling(claim: CanonicalClaim) -> dict[str, Any]:
    b = claim.billing
    s = claim.subscriber
    return {
        "eBilling.01": s.payer_id,
        "eBilling.02": s.payer_name,
        "eBilling.03": s.plan_type,
        "eBilling.07": s.member_id,
        "eBilling.08": s.group_id,
        "eBilling.icd10_diagnoses": [
            {"code": d.code, "display": d.display} for d in b.icd10_diagnoses
        ],
        "eBilling.service_lines": [
            {
                "procedure_code": sl.procedure_code,
                "modifiers": sl.modifiers,
                "units": sl.units,
            }
            for sl in b.service_lines
        ],
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_required(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    t = claim.timeline
    if t.psap_call_datetime is None:
        missing.append("eTimes.01 (psap_call_datetime)")
    if t.unit_on_scene_datetime is None:
        missing.append("eTimes.05 (unit_on_scene_datetime)")
    if t.transfer_of_care_datetime is None:
        missing.append("eTimes.12 (transfer_of_care_datetime)")
    if claim.patient.sex_at_birth is None:
        missing.append("ePatient.13 (sex_at_birth)")
    if claim.clinical.primary_impression is None:
        missing.append("eSituation.11 (primary_impression)")
    if claim.transport.disposition_status is None:
        missing.append("eDisposition.27 (disposition_status)")
    return missing


def _check_optional(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    if not claim.patient.full_name:
        missing.append("ePatient.02 (patient full_name)")
    if claim.patient.age_years is None:
        missing.append("ePatient.16 (age_years)")
    if not claim.clinical.vitals:
        missing.append("eVitals (no vitals recorded)")
    if not claim.clinical.procedures:
        missing.append("eProcedures (no procedures recorded)")
    if not claim.clinical.medications:
        missing.append("eMedications (no medications recorded)")
    if claim.timeline.transport_begin_datetime is None:
        missing.append("eTimes.09 (transport_begin_datetime)")
    return missing


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def export_nemsis(claim: CanonicalClaim) -> ExportResult:
    """Export a ``CanonicalClaim`` to a NEMSIS v3.5 data-element dict.

    Args:
        claim: The v1.0 canonical claim to export.

    Returns:
        An ``ExportResult`` with NEMSIS element keys and validation diagnostics.
    """
    data: dict[str, Any] = {
        "eResponse": _export_eresponse(claim),
        "eTimes": _export_etimes(claim),
        "ePatient": _export_epatient(claim),
        "eSituation": _export_esituation(claim),
        "eVitals": _export_evitals(claim),
        "eProcedures": _export_eprocedures(claim),
        "eMedications": _export_emedications(claim),
        "eDisposition": _export_edisposition(claim),
        "eBilling": _export_ebilling(claim),
        "_meta": {
            "source_claim_id": claim.claim_id,
            "schema_version": claim.schema_version,
            "nemsis_version": "3.5",
        },
    }

    warnings: list[str] = []
    if claim.clinical.primary_impression and claim.clinical.primary_impression.code == "UNKNOWN":
        warnings.append(
            "eSituation.11: primary_impression code is placeholder 'UNKNOWN' — "
            "requires manual ICD-10-CM mapping"
        )

    return ExportResult(
        format="nemsis",
        data=data,
        missing_required=_check_required(claim),
        missing_optional=_check_optional(claim),
        warnings=warnings,
    )
