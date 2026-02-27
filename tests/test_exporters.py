"""Tests for NEMSIS, X12 837, FHIR, and coverage exporters.

Verifies:
- Output structure (required keys present).
- Required-field validation gap detection (missing_required populated).
- Useful missing-field diagnostics (missing_optional, warnings).
- Coverage report structure and machine-readability.
"""

from __future__ import annotations

import json
from pathlib import Path

from ems_pipeline.claim.canonical import (
    BillingInfo,
    CanonicalClaim,
    ClinicalInfo,
    CodedValue,
    EncounterInfo,
    PatientInfo,
    ServiceLine,
    SubscriberInfo,
    TransportInfo,
    VitalSign,
    Medication,
    Procedure,
)
from ems_pipeline.exporters import ExportResult
from ems_pipeline.exporters.coverage import generate_coverage_report
from ems_pipeline.exporters.fhir import export_fhir
from ems_pipeline.exporters.nemsis import export_nemsis
from ems_pipeline.exporters.x12_837 import export_x12_837

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _minimal_claim() -> CanonicalClaim:
    """Absolute minimum claim — most fields missing."""
    return CanonicalClaim(claim_id="min-001")


def _rich_claim() -> CanonicalClaim:
    """Realistic ALS transport claim with most fields populated."""
    return CanonicalClaim(
        claim_id="rich-001",
        patient=PatientInfo(
            full_name="John Smith",
            sex_at_birth="M",
            age_years=54,
        ),
        subscriber=SubscriberInfo(
            member_id="M123456",
            group_id="G789",
            payer_id="PAYER01",
            payer_name="Acme Insurance",
            relationship_to_patient="self",
            plan_type="commercial",
        ),
        encounter=EncounterInfo(
            encounter_id="rich-001",
            run_number="RUN-001",
            agency_id="AG-001",
            agency_name="Metro EMS",
        ),
        clinical=ClinicalInfo(
            chief_complaint="chest pain",
            primary_impression=CodedValue(
                code="R07.9",
                code_system="ICD-10-CM",
                display="Chest pain, unspecified",
                confidence=0.9,
                source="extracted",
            ),
            symptoms=["shortness of breath"],
            vitals=[
                VitalSign(name="BP", value="150/90", unit="mmHg", evidence_segment_ids=["seg_0003"]),
                VitalSign(name="SpO2", value="92%", unit="%", evidence_segment_ids=["seg_0003"]),
            ],
            medications=[
                Medication(drug="aspirin", dose="324 mg", route="oral", evidence_segment_ids=["seg_0004"])
            ],
            procedures=[
                Procedure(name="oxygen administration", evidence_segment_ids=["seg_0004"])
            ],
        ),
        transport=TransportInfo(
            destination_facility_name="Mercy Hospital",
            disposition_status="transported",
        ),
        billing=BillingInfo(
            claim_type="professional",
            place_of_service="41",
            service_lines=[
                ServiceLine(
                    line_number=1,
                    procedure_code="A0427",
                    modifiers=["QM"],
                    diagnosis_pointers=[1],
                    units=1.0,
                    charge_amount=850.00,
                    place_of_service="41",
                )
            ],
            icd10_diagnoses=[
                CodedValue(code="R07.9", code_system="ICD-10-CM", display="Chest pain, unspecified")
            ],
            billing_provider_npi="1234567890",
            billing_provider_name="Metro EMS Agency",
        ),
    )


def _fixture_claim() -> CanonicalClaim:
    """Load the sample fixture."""
    raw = json.loads(
        (FIXTURES_DIR / "canonical_claim_sample.json").read_text(encoding="utf-8")
    )
    return CanonicalClaim.model_validate(raw)


# ---------------------------------------------------------------------------
# ExportResult contract
# ---------------------------------------------------------------------------


def test_export_result_is_valid_when_no_missing_required() -> None:
    result = ExportResult(format="test", data={"ok": True}, missing_required=[])
    assert result.is_valid is True


def test_export_result_invalid_when_missing_required() -> None:
    result = ExportResult(
        format="test",
        data={},
        missing_required=["some.field"],
    )
    assert result.is_valid is False


def test_export_result_to_dict_structure() -> None:
    result = ExportResult(
        format="nemsis",
        data={"key": "val"},
        missing_required=["a"],
        missing_optional=["b"],
        warnings=["w"],
    )
    d = result.to_dict()
    assert d["format"] == "nemsis"
    assert d["is_valid"] is False
    assert "missing_required" in d
    assert "missing_optional" in d
    assert "warnings" in d
    assert "data" in d


# ===========================================================================
# NEMSIS exporter
# ===========================================================================


class TestNemsis:
    def test_returns_export_result(self) -> None:
        result = export_nemsis(_minimal_claim())
        assert isinstance(result, ExportResult)
        assert result.format == "nemsis"

    def test_required_sections_present(self) -> None:
        result = export_nemsis(_rich_claim())
        for section in ("eResponse", "eTimes", "ePatient", "eSituation", "eDisposition"):
            assert section in result.data, f"Missing section: {section}"

    def test_epatient_sex_mapped(self) -> None:
        result = export_nemsis(_rich_claim())
        # "M" → NEMSIS code 9906001
        assert result.data["ePatient"]["ePatient.13"] == "9906001"

    def test_epatient_sex_unknown_for_minimal(self) -> None:
        result = export_nemsis(_minimal_claim())
        # No sex → falls back to unknown code 9906009
        assert result.data["ePatient"]["ePatient.13"] == "9906009"

    def test_esituation_chief_complaint(self) -> None:
        result = export_nemsis(_rich_claim())
        assert result.data["eSituation"]["eSituation.04"] == "chest pain"

    def test_esituation_impression_code(self) -> None:
        result = export_nemsis(_rich_claim())
        assert result.data["eSituation"]["eSituation.11"] == "R07.9"

    def test_evitals_populated(self) -> None:
        result = export_nemsis(_rich_claim())
        vitals = result.data.get("eVitals", [])
        assert len(vitals) == 2

    def test_emedications_populated(self) -> None:
        result = export_nemsis(_rich_claim())
        meds = result.data.get("eMedications", [])
        assert len(meds) == 1
        assert meds[0]["eMedications.03_display"] == "aspirin"

    def test_eprocedures_populated(self) -> None:
        result = export_nemsis(_rich_claim())
        procs = result.data.get("eProcedures", [])
        assert len(procs) == 1

    def test_edisposition_transport_code(self) -> None:
        result = export_nemsis(_rich_claim())
        # "transported" → NEMSIS code 4227001
        assert result.data["eDisposition"]["eDisposition.27"] == "4227001"

    def test_missing_required_sex_at_birth(self) -> None:
        result = export_nemsis(_minimal_claim())
        assert any("sex_at_birth" in f for f in result.missing_required)

    def test_missing_required_primary_impression(self) -> None:
        result = export_nemsis(_minimal_claim())
        assert any("primary_impression" in f for f in result.missing_required)

    def test_missing_required_timestamps(self) -> None:
        result = export_nemsis(_minimal_claim())
        # Should flag missing eTimes fields
        timeline_missing = [f for f in result.missing_required if "eTimes" in f or "datetime" in f]
        assert len(timeline_missing) > 0

    def test_warning_for_unknown_code(self) -> None:
        claim = _rich_claim()
        # Replace impression with placeholder code
        claim = claim.model_copy(
            update={
                "clinical": claim.clinical.model_copy(
                    update={
                        "primary_impression": CodedValue(
                            code="UNKNOWN",
                            code_system="ICD-10-CM",
                            display="chest pain",
                        )
                    }
                )
            }
        )
        result = export_nemsis(claim)
        assert any("UNKNOWN" in w for w in result.warnings)

    def test_meta_block_present(self) -> None:
        result = export_nemsis(_rich_claim())
        assert "_meta" in result.data
        assert result.data["_meta"]["nemsis_version"] == "3.5"

    def test_fixture_claim_exports(self) -> None:
        result = export_nemsis(_fixture_claim())
        assert isinstance(result, ExportResult)
        assert "eResponse" in result.data


# ===========================================================================
# X12 837P exporter
# ===========================================================================


class TestX12837:
    def test_returns_export_result(self) -> None:
        result = export_x12_837(_minimal_claim())
        assert isinstance(result, ExportResult)
        assert result.format == "x12_837"

    def test_transaction_set_header(self) -> None:
        result = export_x12_837(_rich_claim())
        assert result.data["transaction_set"] == "837"
        assert "005010X222A2" in result.data.get("implementation_convention", "")

    def test_2000a_billing_provider_present(self) -> None:
        result = export_x12_837(_rich_claim())
        loop = result.data["loop_2000a_billing_provider"]
        assert "NM1_85" in loop
        assert loop["NM1_85"]["id"] == "1234567890"

    def test_2000b_subscriber_present(self) -> None:
        result = export_x12_837(_rich_claim())
        loop = result.data["loop_2000b_subscriber"]
        assert "NM1_IL" in loop
        assert loop["NM1_IL"]["id"] == "M123456"

    def test_clm_segment_claim_id(self) -> None:
        result = export_x12_837(_rich_claim())
        clm = result.data["loop_2300_claim"]["CLM"]
        assert clm["clm01_claim_id"] == "rich-001"

    def test_hi_diagnoses_present(self) -> None:
        result = export_x12_837(_rich_claim())
        hi = result.data["loop_2300_claim"]["HI"]
        assert len(hi) == 1
        assert hi[0]["code"] == "R07.9"

    def test_service_lines_present(self) -> None:
        result = export_x12_837(_rich_claim())
        lines = result.data["loop_2400_service_lines"]
        assert len(lines) == 1
        sv1 = lines[0]["SV1"]
        assert sv1["procedure_code"] == "A0427"

    def test_missing_required_billing_npi(self) -> None:
        result = export_x12_837(_minimal_claim())
        assert any("billing_provider_npi" in f for f in result.missing_required)

    def test_missing_required_member_id(self) -> None:
        result = export_x12_837(_minimal_claim())
        assert any("member_id" in f for f in result.missing_required)

    def test_missing_required_icd10(self) -> None:
        result = export_x12_837(_minimal_claim())
        assert any("icd10" in f.lower() for f in result.missing_required)

    def test_missing_required_service_lines(self) -> None:
        result = export_x12_837(_minimal_claim())
        assert any("service_line" in f.lower() for f in result.missing_required)

    def test_warning_no_charge_amount(self) -> None:
        claim = _rich_claim()
        claim = claim.model_copy(
            update={
                "billing": claim.billing.model_copy(
                    update={
                        "service_lines": [
                            ServiceLine(line_number=1, procedure_code="A0427")
                        ]
                    }
                )
            }
        )
        result = export_x12_837(claim)
        assert any("charge_amount" in w for w in result.warnings)

    def test_fixture_claim_exports(self) -> None:
        result = export_x12_837(_fixture_claim())
        assert isinstance(result, ExportResult)
        assert "loop_2300_claim" in result.data


# ===========================================================================
# FHIR R4 exporter
# ===========================================================================


class TestFHIR:
    def test_returns_export_result(self) -> None:
        result = export_fhir(_minimal_claim())
        assert isinstance(result, ExportResult)
        assert result.format == "fhir"

    def test_bundle_type(self) -> None:
        result = export_fhir(_rich_claim())
        assert result.data["resourceType"] == "Bundle"
        assert result.data["type"] == "collection"

    def test_patient_resource_present(self) -> None:
        result = export_fhir(_rich_claim())
        patient = next(
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Patient"
        )
        assert patient["name"][0]["family"] == "Smith"
        assert patient["gender"] == "male"

    def test_patient_age_estimate_extension(self) -> None:
        """When DOB is absent but age_years is set, extension is added."""
        result = export_fhir(_rich_claim())
        patient = next(
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Patient"
        )
        # DOB is None, age_years=54 → _birthDate extension
        assert "_birthDate" in patient
        ext = patient["_birthDate"]["extension"][0]
        assert ext["valueInteger"] == 54

    def test_coverage_resource_present(self) -> None:
        result = export_fhir(_rich_claim())
        coverage = next(
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Coverage"
        )
        assert coverage["subscriberId"] == "M123456"

    def test_encounter_resource_present(self) -> None:
        result = export_fhir(_rich_claim())
        enc = next(
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Encounter"
        )
        assert enc["status"] == "finished"
        assert enc["class"]["code"] == "EMER"

    def test_claim_resource_present(self) -> None:
        result = export_fhir(_rich_claim())
        cr = next(
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Claim"
        )
        assert cr["status"] == "active"
        assert cr["use"] == "claim"

    def test_observations_for_each_vital(self) -> None:
        result = export_fhir(_rich_claim())
        obs = [
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Observation"
        ]
        assert len(obs) == 2

    def test_observation_has_loinc_for_bp(self) -> None:
        result = export_fhir(_rich_claim())
        obs = [
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Observation"
        ]
        bp_obs = next(
            o for o in obs
            if any(c.get("code") == "55284-4" for c in o["code"]["coding"])
        )
        assert bp_obs is not None

    def test_procedure_resources_present(self) -> None:
        result = export_fhir(_rich_claim())
        procs = [
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "Procedure"
        ]
        assert len(procs) == 1

    def test_medication_admin_resources_present(self) -> None:
        result = export_fhir(_rich_claim())
        meds = [
            e["resource"]
            for e in result.data["entry"]
            if e["resource"]["resourceType"] == "MedicationAdministration"
        ]
        assert len(meds) == 1
        assert "aspirin" in meds[0]["medicationCodeableConcept"]["text"].lower()

    def test_missing_required_patient_name(self) -> None:
        result = export_fhir(_minimal_claim())
        assert any("full_name" in f for f in result.missing_required)

    def test_missing_required_insurer(self) -> None:
        result = export_fhir(_minimal_claim())
        assert any("payer" in f.lower() for f in result.missing_required)

    def test_warning_for_empty_diagnoses(self) -> None:
        result = export_fhir(_minimal_claim())
        assert any("ICD-10" in w or "diagnosis" in w.lower() for w in result.warnings)

    def test_meta_block_fhir_version(self) -> None:
        result = export_fhir(_rich_claim())
        assert result.data["_meta"]["fhir_version"] == "R4"

    def test_fixture_claim_exports(self) -> None:
        result = export_fhir(_fixture_claim())
        assert isinstance(result, ExportResult)
        assert result.data["resourceType"] == "Bundle"


# ===========================================================================
# Coverage report
# ===========================================================================


class TestCoverageReport:
    def test_returns_dict(self) -> None:
        report = generate_coverage_report(_minimal_claim())
        assert isinstance(report, dict)

    def test_top_level_keys_present(self) -> None:
        report = generate_coverage_report(_rich_claim())
        assert "schema_version" in report
        assert "claim_id" in report
        assert "canonical_field_summary" in report
        assert "formats" in report

    def test_formats_keys(self) -> None:
        report = generate_coverage_report(_rich_claim())
        assert set(report["formats"].keys()) == {"nemsis", "x12_837", "fhir"}

    def test_each_format_has_required_keys(self) -> None:
        report = generate_coverage_report(_rich_claim())
        for fmt_name, fmt_data in report["formats"].items():
            assert "is_valid" in fmt_data, f"{fmt_name} missing is_valid"
            assert "missing_required" in fmt_data, f"{fmt_name} missing missing_required"
            assert "missing_optional" in fmt_data, f"{fmt_name} missing missing_optional"
            assert "warnings" in fmt_data, f"{fmt_name} missing warnings"

    def test_canonical_field_summary_coverage_string(self) -> None:
        report = generate_coverage_report(_rich_claim())
        summary = report["canonical_field_summary"]
        assert "coverage" in summary
        assert "/" in summary["coverage"]  # e.g. "12/45 (26%)"

    def test_minimal_claim_has_missing_fields(self) -> None:
        report = generate_coverage_report(_minimal_claim())
        summary = report["canonical_field_summary"]
        # Most fields should be missing for a minimal claim
        assert len(summary["missing"]) > len(summary["filled"])

    def test_rich_claim_has_more_filled_fields(self) -> None:
        min_report = generate_coverage_report(_minimal_claim())
        rich_report = generate_coverage_report(_rich_claim())
        assert len(rich_report["canonical_field_summary"]["filled"]) > len(
            min_report["canonical_field_summary"]["filled"]
        )

    def test_report_is_json_serializable(self) -> None:
        report = generate_coverage_report(_rich_claim())
        serialized = json.dumps(report, default=str)
        assert json.loads(serialized) is not None

    def test_fixture_claim_coverage(self) -> None:
        report = generate_coverage_report(_fixture_claim())
        assert report["claim_id"] == _fixture_claim().claim_id
