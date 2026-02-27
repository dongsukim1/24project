"""Unit tests for canonical claim model (schema_version=1.0).

Tests validate:
- Model construction with all fields.
- Default factory values (empty objects / lists).
- Field constraints (confidence bounds, literal sex_at_birth).
- JSON round-trip via model_dump / model_validate.
- Fixture file loads correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ems_pipeline.claim.canonical import (
    Address,
    BillingInfo,
    CanonicalClaim,
    CanonicalProvenance,
    ClinicalInfo,
    CodedValue,
    CrewMember,
    EncounterInfo,
    IncidentTimeline,
    Medication,
    PatientInfo,
    Procedure,
    ServiceLine,
    SubscriberInfo,
    TransportInfo,
    UnitInfo,
    VitalSign,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _minimal_claim(claim_id: str = "test-claim-001") -> CanonicalClaim:
    return CanonicalClaim(claim_id=claim_id)


# ---------------------------------------------------------------------------
# CodedValue
# ---------------------------------------------------------------------------


def test_coded_value_required_fields() -> None:
    cv = CodedValue(code="I21.9", code_system="ICD-10-CM")
    assert cv.code == "I21.9"
    assert cv.code_system == "ICD-10-CM"
    assert cv.display is None
    assert cv.confidence is None
    assert cv.source is None


def test_coded_value_full() -> None:
    cv = CodedValue(
        code="A0427",
        code_system="HCPCS",
        display="Ambulance ALS emergency",
        confidence=0.95,
        source="agent2",
    )
    assert cv.display == "Ambulance ALS emergency"
    assert cv.confidence == 0.95


def test_coded_value_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        CodedValue(code="X", code_system="Y", confidence=1.5)
    with pytest.raises(ValidationError):
        CodedValue(code="X", code_system="Y", confidence=-0.1)


def test_coded_value_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        CodedValue(code="X", code_system="Y", unknown_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PatientInfo
# ---------------------------------------------------------------------------


def test_patient_info_defaults() -> None:
    p = PatientInfo()
    assert p.full_name is None
    assert p.sex_at_birth is None
    assert p.race == []
    assert p.age_years is None


def test_patient_info_sex_at_birth_valid() -> None:
    for sex in ("M", "F", "U"):
        p = PatientInfo(sex_at_birth=sex)
        assert p.sex_at_birth == sex


def test_patient_info_sex_at_birth_invalid() -> None:
    with pytest.raises(ValidationError):
        PatientInfo(sex_at_birth="X")  # type: ignore[arg-type]


def test_patient_info_address() -> None:
    p = PatientInfo(address=Address(street="123 Main", city="Anytown", state="CA"))
    assert p.address is not None
    assert p.address.city == "Anytown"


# ---------------------------------------------------------------------------
# VitalSign / Medication / Procedure
# ---------------------------------------------------------------------------


def test_vital_sign_required() -> None:
    v = VitalSign(name="BP", value="120/80")
    assert v.unit is None
    assert v.evidence_segment_ids == []


def test_vital_sign_full() -> None:
    v = VitalSign(name="SpO2", value="98", unit="%", evidence_segment_ids=["seg_0003"])
    assert v.unit == "%"
    assert v.evidence_segment_ids == ["seg_0003"]


def test_medication_required() -> None:
    m = Medication(drug="aspirin")
    assert m.dose is None
    assert m.route is None
    assert m.coded is None


def test_procedure_required() -> None:
    p = Procedure(name="IV access")
    assert p.coded is None
    assert p.evidence_segment_ids == []


# ---------------------------------------------------------------------------
# ServiceLine / BillingInfo
# ---------------------------------------------------------------------------


def test_service_line_defaults() -> None:
    sl = ServiceLine(line_number=1, procedure_code="A0427")
    assert sl.units == 1.0
    assert sl.modifiers == []
    assert sl.diagnosis_pointers == []


def test_billing_info_defaults() -> None:
    b = BillingInfo()
    assert b.service_lines == []
    assert b.icd10_diagnoses == []
    assert b.claim_type is None


# ---------------------------------------------------------------------------
# CanonicalClaim defaults and structure
# ---------------------------------------------------------------------------


def test_canonical_claim_minimal() -> None:
    claim = _minimal_claim()
    assert claim.schema_version == "1.0"
    assert claim.claim_id == "test-claim-001"
    assert claim.patient == PatientInfo()
    assert claim.provenance == []
    assert claim.legacy_claim_id is None


def test_canonical_claim_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        CanonicalClaim(claim_id="x", unknown_field="y")  # type: ignore[call-arg]


def test_canonical_claim_schema_version_preserved() -> None:
    claim = CanonicalClaim(claim_id="x", schema_version="1.0")
    assert claim.schema_version == "1.0"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_canonical_provenance_required() -> None:
    p = CanonicalProvenance(segment_id="seg_0000")
    assert p.entity_index is None
    assert p.field_path is None


def test_canonical_provenance_entity_index_non_negative() -> None:
    with pytest.raises(ValidationError):
        CanonicalProvenance(segment_id="seg_0000", entity_index=-1)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_canonical_claim_json_roundtrip() -> None:
    claim = CanonicalClaim(
        claim_id="rt-001",
        patient=PatientInfo(full_name="Jane Doe", sex_at_birth="F", age_years=34),
        clinical=ClinicalInfo(
            chief_complaint="shortness of breath",
            vitals=[VitalSign(name="SpO2", value="94", unit="%")],
            medications=[Medication(drug="albuterol", dose="2.5mg", route="inhaled")],
        ),
        transport=TransportInfo(
            destination_facility_name="City Hospital",
            disposition_status="transported",
        ),
        billing=BillingInfo(
            service_lines=[ServiceLine(line_number=1, procedure_code="A0427")],
            icd10_diagnoses=[CodedValue(code="R06.0", code_system="ICD-10-CM")],
        ),
    )
    dumped = claim.model_dump(mode="json")
    restored = CanonicalClaim.model_validate(dumped)
    assert restored.claim_id == claim.claim_id
    assert restored.patient.full_name == "Jane Doe"
    assert restored.clinical.vitals[0].name == "SpO2"
    assert restored.clinical.medications[0].drug == "albuterol"
    assert restored.transport.disposition_status == "transported"
    assert restored.billing.service_lines[0].procedure_code == "A0427"
    assert restored.billing.icd10_diagnoses[0].code == "R06.0"


# ---------------------------------------------------------------------------
# Fixture file
# ---------------------------------------------------------------------------


def test_canonical_claim_fixture_loads() -> None:
    fixture_path = FIXTURES_DIR / "canonical_claim_sample.json"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    claim = CanonicalClaim.model_validate(raw)
    assert claim.schema_version == "1.0"
    assert claim.patient.sex_at_birth == "M"
    assert claim.patient.age_years == 54
    assert claim.clinical.chief_complaint == "chest pain"
    assert len(claim.clinical.vitals) == 2
    assert len(claim.clinical.medications) == 1
    assert claim.transport.destination_facility_name == "Mercy Hospital"
    assert claim.transport.disposition_status == "transported"
    assert len(claim.billing.service_lines) == 1
    assert claim.billing.service_lines[0].procedure_code == "A0427"
    assert len(claim.billing.icd10_diagnoses) == 1
    assert claim.billing.icd10_diagnoses[0].code == "R07.9"
    assert len(claim.provenance) > 0
