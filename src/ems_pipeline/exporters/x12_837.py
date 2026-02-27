"""X12 837P professional claim exporter.

Maps a ``CanonicalClaim`` to a structured Python dict that mirrors the X12 837P
transaction set layout.  This is *not* a raw EDI segment string generator; the
dict structure is designed to be:

1. Human-readable for review and testing.
2. Straightforwardly convertible to an EDI string by a downstream formatter.

X12 837P loop/segment reference (simplified):
    ISA  — Interchange Control Header (handled by EDI envelope layer)
    GS   — Functional Group Header
    ST   — Transaction Set Header (837)
    BPR  — Beginning of Payment (for 835; informational here)
    NM1*41 — Submitter name
    NM1*40 — Receiver name
    2000A  — Billing/Pay-To Provider Hierarchical Loop
      NM1*85 — Billing provider
      REF*EI — Federal Tax ID
    2000B  — Subscriber Hierarchical Loop
      NM1*IL — Subscriber
      DMG    — Subscriber demographics
    2000C  — Patient Hierarchical Loop (when patient ≠ subscriber)
      NM1*QC — Patient name
      DMG    — Patient demographics
    2300   — Claim Information
      CLM   — Claim
      DTP*472 — Service date
      HI    — Diagnoses (ICD-10-CM)
      2400  — Service Line
        LX  — Line counter
        SV1 — Professional service
        DTP*472 — Service date

Covered payer types: commercial / Medicare / Medicaid (structural; no payer-
specific edits in MVP).

ASC X12N 837P Implementation Guide reference:
    https://x12.org/codes/transaction-sets
"""

from __future__ import annotations

from typing import Any

from ems_pipeline.claim.canonical import CanonicalClaim
from ems_pipeline.exporters import ExportResult

# ---------------------------------------------------------------------------
# Required fields for a valid 837P (MVP subset)
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "billing_provider_npi",
    "subscriber.member_id",
    "patient.full_name",
    "claim.clm01_claim_id",
    "claim.clm05_place_of_service",
    "icd10_diagnoses (at least one)",
    "service_lines (at least one)",
]

_OPTIONAL_FIELDS = [
    "patient.dob",
    "patient.sex_at_birth",
    "rendering_provider_npi",
    "billing_provider_tax_id",
    "service_date",
]


# ---------------------------------------------------------------------------
# Value maps
# ---------------------------------------------------------------------------

_SEX_MAP = {"M": "M", "F": "F", "U": "U"}

_PLAN_TYPE_MAP = {
    "medicare":   "MB",
    "medicaid":   "MC",
    "commercial": "BL",  # Blue Cross Blue Shield as fallback
}


def _fmt_date(dt: Any) -> str | None:
    """Format a date or datetime as YYYYMMDD for X12."""
    if dt is None:
        return None
    try:
        return dt.strftime("%Y%m%d")
    except AttributeError:
        return str(dt)[:10].replace("-", "")


def _fmt_name(full_name: str | None) -> tuple[str | None, str | None]:
    """Split 'First Last' → (last, first)."""
    if not full_name:
        return None, None
    parts = full_name.strip().split(" ", 1)
    if len(parts) == 2:
        return parts[1], parts[0]
    return parts[0], None


# ---------------------------------------------------------------------------
# Loop / segment builders
# ---------------------------------------------------------------------------


def _build_2000a(claim: CanonicalClaim) -> dict[str, Any]:
    """Loop 2000A – Billing Provider."""
    b = claim.billing
    return {
        "NM1_85": {
            "entity_id": "85",
            "entity_type": "2",  # non-person
            "org_name": b.billing_provider_name,
            "id_qualifier": "XX",
            "id": b.billing_provider_npi,
        },
        "REF_EI": {
            "qualifier": "EI",
            "value": b.billing_provider_tax_id,
        },
    }


def _build_2000b(claim: CanonicalClaim) -> dict[str, Any]:
    """Loop 2000B – Subscriber."""
    s = claim.subscriber
    p = claim.patient
    last, first = _fmt_name(s.subscriber_name or p.full_name)
    plan_code = _PLAN_TYPE_MAP.get((s.plan_type or "").lower(), "ZZ")
    return {
        "SBR": {
            "payer_responsibility": "P",  # Primary
            "relationship_code": "18" if s.relationship_to_patient in ("self", None) else "01",
            "member_id": s.member_id,
            "plan_type": plan_code,
        },
        "NM1_IL": {
            "entity_id": "IL",
            "entity_type": "1",
            "last_name": last,
            "first_name": first,
            "id_qualifier": "MI",
            "id": s.member_id,
        },
        "NM1_PR": {
            "entity_id": "PR",
            "entity_type": "2",
            "org_name": s.payer_name,
            "id_qualifier": "PI",
            "id": s.payer_id,
        },
    }


def _build_2000c(claim: CanonicalClaim) -> dict[str, Any] | None:
    """Loop 2000C – Patient (only when patient ≠ subscriber)."""
    p = claim.patient
    s = claim.subscriber
    if s.relationship_to_patient in ("self", None) and not p.full_name:
        return None
    last, first = _fmt_name(p.full_name)
    dob = _fmt_date(p.dob)
    sex = _SEX_MAP.get(p.sex_at_birth or "", "U")
    return {
        "PAT": {
            "relationship_code": "19",  # Child (default fallback)
        },
        "NM1_QC": {
            "entity_id": "QC",
            "entity_type": "1",
            "last_name": last,
            "first_name": first,
        },
        "DMG": {
            "date_format": "D8",
            "dob": dob,
            "sex": sex,
        },
    }


def _build_clm(claim: CanonicalClaim) -> dict[str, Any]:
    """Loop 2300 – Claim Information (CLM segment)."""
    b = claim.billing
    svc_date = (
        _fmt_date(claim.encounter.service_date)
        or _fmt_date(claim.encounter.service_start_datetime)
    )
    pos = b.place_of_service or "41"
    total_charge = sum(
        (sl.charge_amount or 0.0) for sl in b.service_lines
    )
    return {
        "CLM": {
            "clm01_claim_id": claim.claim_id,
            "clm02_total_charge": round(total_charge, 2) if total_charge else None,
            "clm05_facility_code": pos,
            "clm05_claim_type": b.claim_type or "professional",
            "clm11_assignment": "Y",
        },
        "DTP_472": {
            "qualifier": "472",
            "format": "D8",
            "date": svc_date,
        },
        "HI": [
            {
                "qualifier": "ABK",  # ICD-10-CM Principal Diagnosis
                "code": dx.code,
                "display": dx.display,
            }
            for dx in b.icd10_diagnoses
        ],
        "NM1_82": {
            "entity_id": "82",
            "entity_type": "1",
            "id_qualifier": "XX",
            "id": b.rendering_provider_npi,
        },
    }


def _build_service_lines(claim: CanonicalClaim) -> list[dict[str, Any]]:
    """Loop 2400 – Service Lines."""
    svc_date = (
        _fmt_date(claim.encounter.service_date)
        or _fmt_date(claim.encounter.service_start_datetime)
    )
    out: list[dict[str, Any]] = []
    for sl in claim.billing.service_lines:
        out.append({
            "LX": sl.line_number,
            "SV1": {
                "procedure_code": sl.procedure_code,
                "modifiers": sl.modifiers,
                "charge_amount": sl.charge_amount,
                "units": sl.units,
                "place_of_service": sl.place_of_service or claim.billing.place_of_service or "41",
                "diagnosis_pointers": sl.diagnosis_pointers,
            },
            "DTP_472": {
                "qualifier": "472",
                "format": "D8",
                "date": svc_date,
            },
        })
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_required(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    if not claim.billing.billing_provider_npi:
        missing.append("billing.billing_provider_npi")
    if not claim.subscriber.member_id:
        missing.append("subscriber.member_id")
    if not claim.patient.full_name:
        missing.append("patient.full_name")
    if not claim.billing.icd10_diagnoses:
        missing.append("billing.icd10_diagnoses (at least one ICD-10 required)")
    if not claim.billing.service_lines:
        missing.append("billing.service_lines (at least one CPT/HCPCS service line required)")
    return missing


def _check_optional(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    if claim.patient.dob is None and claim.patient.age_years is None:
        missing.append("patient.dob or patient.age_years")
    if claim.patient.sex_at_birth is None:
        missing.append("patient.sex_at_birth")
    if not claim.billing.rendering_provider_npi:
        missing.append("billing.rendering_provider_npi")
    if not claim.billing.billing_provider_tax_id:
        missing.append("billing.billing_provider_tax_id")
    svc_date = claim.encounter.service_date or claim.encounter.service_start_datetime
    if svc_date is None:
        missing.append("encounter.service_date or service_start_datetime")
    return missing


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def export_x12_837(claim: CanonicalClaim) -> ExportResult:
    """Export a ``CanonicalClaim`` to an X12 837P structured representation.

    Args:
        claim: The v1.0 canonical claim to export.

    Returns:
        An ``ExportResult`` with the 837P loop/segment dict and validation
        diagnostics.
    """
    patient_loop = _build_2000c(claim)

    data: dict[str, Any] = {
        "transaction_set": "837",
        "implementation_convention": "005010X222A2",
        "loop_2000a_billing_provider": _build_2000a(claim),
        "loop_2000b_subscriber": _build_2000b(claim),
        "loop_2300_claim": _build_clm(claim),
        "loop_2400_service_lines": _build_service_lines(claim),
        "_meta": {
            "source_claim_id": claim.claim_id,
            "schema_version": claim.schema_version,
        },
    }
    if patient_loop is not None:
        data["loop_2000c_patient"] = patient_loop

    warnings: list[str] = []
    if not claim.billing.service_lines:
        warnings.append(
            "No CPT/HCPCS service lines present — "
            "the 837P requires at least one SV1 segment"
        )
    for sl in claim.billing.service_lines:
        if sl.charge_amount is None:
            warnings.append(
                f"Service line {sl.line_number} ({sl.procedure_code}) has no "
                "charge_amount — CLM02 total charge will be 0"
            )

    return ExportResult(
        format="x12_837",
        data=data,
        missing_required=_check_required(claim),
        missing_optional=_check_optional(claim),
        warnings=warnings,
    )
