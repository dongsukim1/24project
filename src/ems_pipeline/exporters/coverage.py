"""Cross-format field coverage reporter.

Generates a machine-readable JSON report showing, per export format, which
canonical claim fields are populated versus missing.  Useful for CI checks and
developer feedback.

Usage::

    from ems_pipeline.exporters.coverage import generate_coverage_report

    report = generate_coverage_report(canonical_claim)
    print(json.dumps(report, indent=2))

Output structure::

    {
      "schema_version": "1.0",
      "claim_id": "...",
      "canonical_field_summary": { ... },
      "formats": {
        "nemsis":   { "required": [...], "optional": [...], "warnings": [...] },
        "x12_837":  { ... },
        "fhir":     { ... }
      }
    }
"""

from __future__ import annotations

from typing import Any

from ems_pipeline.claim.canonical import CanonicalClaim
from ems_pipeline.exporters.nemsis import export_nemsis
from ems_pipeline.exporters.x12_837 import export_x12_837
from ems_pipeline.exporters.fhir import export_fhir


# ---------------------------------------------------------------------------
# Canonical field summary (format-agnostic)
# ---------------------------------------------------------------------------


def _field_summary(claim: CanonicalClaim) -> dict[str, Any]:
    """Produce a flat presence/absence map of canonical top-level fields."""
    p = claim.patient
    s = claim.subscriber
    e = claim.encounter
    u = claim.unit
    t = claim.timeline
    c = claim.clinical
    tr = claim.transport
    b = claim.billing

    def present(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, (list, dict)):
            return len(v) > 0
        return bool(str(v).strip())

    return {
        # Section A – Patient
        "patient.full_name":       present(p.full_name),
        "patient.dob":             present(p.dob),
        "patient.age_years":       present(p.age_years),
        "patient.sex_at_birth":    present(p.sex_at_birth),
        "patient.gender_identity": present(p.gender_identity),
        "patient.race":            present(p.race),
        "patient.ethnicity":       present(p.ethnicity),
        "patient.address":         present(p.address),
        "patient.phone":           present(p.phone),
        # Section B – Subscriber
        "subscriber.member_id":          present(s.member_id),
        "subscriber.group_id":           present(s.group_id),
        "subscriber.payer_id":           present(s.payer_id),
        "subscriber.payer_name":         present(s.payer_name),
        "subscriber.plan_type":          present(s.plan_type),
        "subscriber.relationship":       present(s.relationship_to_patient),
        # Section C – Encounter
        "encounter.run_number":          present(e.run_number),
        "encounter.cad_incident_number": present(e.cad_incident_number),
        "encounter.pcr_number":          present(e.pcr_number),
        "encounter.agency_id":           present(e.agency_id),
        "encounter.service_date":        present(e.service_date),
        # Section D – Unit
        "unit.unit_id":  present(u.unit_id),
        "unit.crew":     present(u.crew),
        # Section E – Timeline
        "timeline.psap_call":         present(t.psap_call_datetime),
        "timeline.dispatch_notified": present(t.dispatch_notified_datetime),
        "timeline.unit_enroute":      present(t.unit_enroute_datetime),
        "timeline.unit_on_scene":     present(t.unit_on_scene_datetime),
        "timeline.patient_contact":   present(t.patient_contact_datetime),
        "timeline.transport_begin":   present(t.transport_begin_datetime),
        "timeline.arrive_destination":present(t.arrive_destination_datetime),
        "timeline.transfer_of_care":  present(t.transfer_of_care_datetime),
        # Section F – Clinical
        "clinical.chief_complaint":    present(c.chief_complaint),
        "clinical.primary_impression": present(c.primary_impression),
        "clinical.symptoms":           present(c.symptoms),
        "clinical.vitals":             present(c.vitals),
        "clinical.procedures":         present(c.procedures),
        "clinical.medications":        present(c.medications),
        "clinical.allergies":          present(c.allergies),
        "clinical.pmh":                present(c.pmh),
        # Section G – Transport
        "transport.origin_address":       present(tr.origin_address),
        "transport.destination_name":     present(tr.destination_facility_name),
        "transport.destination_facility_id": present(tr.destination_facility_id),
        "transport.disposition_status":   present(tr.disposition_status),
        # Section H – Billing
        "billing.service_lines":          present(b.service_lines),
        "billing.icd10_diagnoses":        present(b.icd10_diagnoses),
        "billing.billing_provider_npi":   present(b.billing_provider_npi),
        "billing.rendering_provider_npi": present(b.rendering_provider_npi),
        "billing.medical_necessity_notes":present(b.medical_necessity_notes),
    }


def _pct(summary: dict[str, Any]) -> str:
    total = len(summary)
    filled = sum(1 for v in summary.values() if v)
    return f"{filled}/{total} ({100 * filled // total}%)" if total else "0/0 (0%)"


# ---------------------------------------------------------------------------
# Per-format coverage summaries
# ---------------------------------------------------------------------------


def _format_entry(result: Any) -> dict[str, Any]:
    return {
        "is_valid": result.is_valid,
        "missing_required": result.missing_required,
        "missing_optional": result.missing_optional,
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_coverage_report(claim: CanonicalClaim) -> dict[str, Any]:
    """Generate a cross-format field coverage report.

    Runs all three exporters against *claim* and collects their validation
    results into a single machine-readable JSON dict.

    Args:
        claim: The ``CanonicalClaim`` (v1.0) to analyse.

    Returns:
        A JSON-serialisable dict suitable for CI field-coverage checks.
    """
    summary = _field_summary(claim)

    nemsis_result = export_nemsis(claim)
    x12_result = export_x12_837(claim)
    fhir_result = export_fhir(claim)

    filled = [k for k, v in summary.items() if v]
    missing = [k for k, v in summary.items() if not v]

    return {
        "schema_version": claim.schema_version,
        "claim_id": claim.claim_id,
        "canonical_field_summary": {
            "coverage": _pct(summary),
            "filled": filled,
            "missing": missing,
            "detail": summary,
        },
        "formats": {
            "nemsis":  _format_entry(nemsis_result),
            "x12_837": _format_entry(x12_result),
            "fhir":    _format_entry(fhir_result),
        },
    }
