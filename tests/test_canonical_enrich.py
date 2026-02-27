"""Tests for the canonical claim enrichment layer (enrich.py).

Tests validate deterministic mapping from transcript + entities → CanonicalClaim.
Uses the same toy fixtures as test_claim_builder.py for consistency.
"""

from __future__ import annotations

from ems_pipeline.claim.enrich import enrich_canonical_claim
from ems_pipeline.claim.timeline import build_events
from ems_pipeline.models import Entity, Segment, Transcript


# ---------------------------------------------------------------------------
# Shared fixtures (inline, no conftest dependency)
# ---------------------------------------------------------------------------


def _toy_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                start=0.0,
                end=3.0,
                speaker="spk0",
                text="00:05 911 call received at 123 Main St.",
                confidence=0.9,
            ),
            Segment(
                start=3.0,
                end=6.0,
                speaker="spk0",
                text="00:10 Unit 12 dispatched priority 1 for chest pain.",
                confidence=0.9,
            ),
            Segment(
                start=60.0,
                end=66.0,
                speaker="spk1",
                text="01:00 Patient is a 54 year old male complaining of chest pain and shortness of breath.",
                confidence=0.9,
            ),
            Segment(
                start=80.0,
                end=84.0,
                speaker="spk1",
                text="01:20 BP 150/90, SpO2 92%.",
                confidence=0.9,
            ),
            Segment(
                start=120.0,
                end=125.0,
                speaker="spk1",
                text="02:00 Administered aspirin 324 mg and started oxygen.",
                confidence=0.9,
            ),
            Segment(
                start=180.0,
                end=184.0,
                speaker="spk0",
                text="03:00 Transporting to Mercy Hospital ED.",
                confidence=0.9,
            ),
        ],
        metadata={
            "audio_filename": "toy.wav",
            "segment_id_map": {
                "seg_0000": 0,
                "seg_0001": 1,
                "seg_0002": 2,
                "seg_0003": 3,
                "seg_0004": 4,
                "seg_0005": 5,
            },
        },
    )


def _toy_entities() -> list[Entity]:
    return [
        Entity(type="location", text="123 Main St", normalized="123 Main St", start=0.5, end=1.0),
        Entity(type="unit_id", text="Unit 12", normalized="UNIT12", start=3.1, end=3.4),
        Entity(type="priority", text="priority 1", normalized="1", start=3.8, end=4.2),
        Entity(type="incident_type", text="chest pain", normalized="chest pain", start=4.6, end=5.1),
        Entity(type="age", text="54 year old", normalized="54", start=61.0, end=61.2, confidence=0.9),
        Entity(type="sex", text="male", normalized="male", start=61.8, end=62.1, confidence=0.9),
        Entity(type="chief_complaint", text="chest pain", normalized="chest pain", start=62.5, end=63.0, confidence=0.9),
        Entity(type="symptom", text="shortness of breath", normalized="SOB", start=63.2, end=64.5),
        Entity(
            type="vital",
            text="BP 150/90",
            normalized="150/90",
            start=80.5,
            end=81.5,
            attributes={"name": "BP", "value": "150/90", "unit": "mmHg"},
        ),
        Entity(
            type="vital",
            text="SpO2 92%",
            normalized="92%",
            start=82.0,
            end=83.0,
            attributes={"name": "SpO2", "value": "92%", "unit": "%"},
        ),
        Entity(type="medication", text="aspirin 324 mg", normalized="aspirin 324 mg", start=121.0, end=121.8),
        Entity(type="procedure", text="oxygen", normalized="oxygen", start=123.0, end=123.5),
        Entity(type="destination", text="Mercy Hospital", normalized="Mercy Hospital", start=181.0, end=182.0),
        Entity(type="disposition", text="transported", normalized="transported", start=180.0, end=184.0),
    ]


def _build_canonical(
    agent2_codes: list[str] | None = None,
    payer_id: str | None = None,
) -> "CanonicalClaim":  # noqa: F821
    from ems_pipeline.claim.enrich import enrich_canonical_claim

    transcript = _toy_transcript()
    entities = _toy_entities()
    events = build_events(transcript, entities)
    return enrich_canonical_claim(
        claim_id="test-001",
        transcript=transcript,
        entities=entities,
        events=events,
        agent2_codes=agent2_codes,
        payer_id=payer_id,
    )


# ---------------------------------------------------------------------------
# Patient section
# ---------------------------------------------------------------------------


def test_patient_age_extracted() -> None:
    claim = _build_canonical()
    assert claim.patient.age_years == 54


def test_patient_sex_normalised_to_M() -> None:
    claim = _build_canonical()
    assert claim.patient.sex_at_birth == "M"


def test_patient_age_hint_text_preserved() -> None:
    claim = _build_canonical()
    # raw text is "54" (the normalized value from the entity)
    assert claim.patient.age_hint_text is not None


# ---------------------------------------------------------------------------
# Clinical section
# ---------------------------------------------------------------------------


def test_chief_complaint_populated() -> None:
    claim = _build_canonical()
    assert claim.clinical.chief_complaint == "chest pain"


def test_primary_impression_populated() -> None:
    claim = _build_canonical()
    assert claim.clinical.primary_impression is not None
    assert claim.clinical.primary_impression.display == "chest pain"
    assert claim.clinical.primary_impression.code_system == "ICD-10-CM"


def test_symptoms_extracted() -> None:
    claim = _build_canonical()
    assert "SOB" in claim.clinical.symptoms


def test_vitals_extracted() -> None:
    claim = _build_canonical()
    names = {v.name for v in claim.clinical.vitals}
    assert "BP" in names
    assert "SpO2" in names


def test_vital_bp_has_unit() -> None:
    claim = _build_canonical()
    bp = next(v for v in claim.clinical.vitals if v.name == "BP")
    assert bp.unit == "mmHg"
    assert bp.value == "150/90"


def test_vital_spo2_has_unit() -> None:
    claim = _build_canonical()
    spo2 = next(v for v in claim.clinical.vitals if v.name == "SpO2")
    assert spo2.unit == "%"


def test_medications_extracted() -> None:
    claim = _build_canonical()
    assert len(claim.clinical.medications) == 1
    assert "aspirin" in claim.clinical.medications[0].drug.lower()


def test_procedures_extracted() -> None:
    claim = _build_canonical()
    assert len(claim.clinical.procedures) == 1
    assert "oxygen" in claim.clinical.procedures[0].name.lower()


# ---------------------------------------------------------------------------
# Transport section
# ---------------------------------------------------------------------------


def test_disposition_status_transported() -> None:
    claim = _build_canonical()
    assert claim.transport.disposition_status == "transported"


def test_destination_name_extracted() -> None:
    claim = _build_canonical()
    assert claim.transport.destination_facility_name == "Mercy Hospital"


def test_origin_address_extracted_from_location() -> None:
    claim = _build_canonical()
    assert claim.transport.origin_address is not None
    assert "123 Main" in (claim.transport.origin_address.street or "")


# ---------------------------------------------------------------------------
# Unit section
# ---------------------------------------------------------------------------


def test_unit_id_extracted() -> None:
    claim = _build_canonical()
    assert claim.unit.unit_id == "UNIT12"


# ---------------------------------------------------------------------------
# Encounter section
# ---------------------------------------------------------------------------


def test_encounter_id_equals_claim_id() -> None:
    claim = _build_canonical()
    assert claim.encounter.encounter_id == "test-001"


def test_encounter_run_number_from_metadata() -> None:
    # Metadata in toy transcript has no run_number → should be None
    claim = _build_canonical()
    assert claim.encounter.run_number is None


# ---------------------------------------------------------------------------
# Subscriber / payer
# ---------------------------------------------------------------------------


def test_payer_id_forwarded() -> None:
    claim = _build_canonical(payer_id="PAYER-XYZ")
    assert claim.subscriber.payer_id == "PAYER-XYZ"


def test_payer_id_none_by_default() -> None:
    claim = _build_canonical()
    assert claim.subscriber.payer_id is None


# ---------------------------------------------------------------------------
# Billing from agent2 codes
# ---------------------------------------------------------------------------


def test_billing_icd10_code_classified() -> None:
    claim = _build_canonical(agent2_codes=["R07.9", "A0427"])
    icd_codes = [d.code for d in claim.billing.icd10_diagnoses]
    cpt_codes = [sl.procedure_code for sl in claim.billing.service_lines]
    assert "R07.9" in icd_codes
    assert "A0427" in cpt_codes


def test_billing_place_of_service_ambulance() -> None:
    claim = _build_canonical(agent2_codes=["A0427"])
    assert claim.billing.place_of_service == "41"
    assert claim.billing.service_lines[0].place_of_service == "41"


def test_billing_empty_when_no_codes() -> None:
    claim = _build_canonical()
    assert claim.billing.service_lines == []
    assert claim.billing.icd10_diagnoses == []


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_populated() -> None:
    claim = _build_canonical()
    assert len(claim.provenance) > 0


def test_provenance_segment_ids_reference_known_segs() -> None:
    claim = _build_canonical()
    known = {f"seg_{i:04d}" for i in range(6)}
    for p in claim.provenance:
        assert p.segment_id in known, f"Unknown segment_id: {p.segment_id}"


def test_provenance_field_paths_present() -> None:
    claim = _build_canonical()
    paths = {p.field_path for p in claim.provenance if p.field_path}
    assert "patient.age_years" in paths
    assert "clinical.chief_complaint" in paths
    assert "clinical.vitals" in paths


# ---------------------------------------------------------------------------
# Schema version and legacy linkage
# ---------------------------------------------------------------------------


def test_schema_version_is_1_0() -> None:
    claim = _build_canonical()
    assert claim.schema_version == "1.0"


def test_legacy_claim_id_none_when_not_provided() -> None:
    claim = _build_canonical()
    assert claim.legacy_claim_id is None


def test_legacy_claim_id_forwarded_when_provided() -> None:
    from ems_pipeline.models import Claim

    legacy = Claim(claim_id="legacy-abc", fields={}, provenance=[])
    transcript = _toy_transcript()
    entities = _toy_entities()
    events = build_events(transcript, entities)
    canonical = enrich_canonical_claim(
        claim_id="test-001",
        transcript=transcript,
        entities=entities,
        events=events,
        legacy_claim=legacy,
    )
    assert canonical.legacy_claim_id == "legacy-abc"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_enrichment_is_deterministic() -> None:
    """Same inputs → same output (excluding UUID-generated IDs)."""
    c1 = _build_canonical()
    c2 = _build_canonical()
    assert c1.model_dump(mode="json") == c2.model_dump(mode="json")
