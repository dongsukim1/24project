"""Canonical EMS claim model (schema_version=1.0).

This module defines the fully-structured, strongly-typed claim schema used for
downstream interoperability exports (NEMSIS, X12 837, FHIR/HL7).

Migration from 0.1:
    The v0.1 ``Claim`` (from ``ems_pipeline.models``) stores everything in a
    free-form ``fields`` dict.  Use ``enrich_canonical_claim()`` from
    ``ems_pipeline.claim.enrich`` to produce a v1.0 ``CanonicalClaim`` from the
    transcript + entities used to build the v0.1 claim.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class CodedValue(BaseModel):
    """A coded clinical or administrative value with code-system attribution.

    Attributes:
        code:        The code string (e.g. "I21.9", "A0425", "9906001").
        code_system: Vocabulary identifier (e.g. "ICD-10-CM", "CPT", "HCPCS",
                     "SNOMED-CT", "NEMSIS", "HL7").
        display:     Human-readable label for the code.
        confidence:  Extraction confidence in [0, 1]; None if not applicable.
        source:      How the code was obtained: "extracted", "inferred", "manual".
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    code_system: str
    display: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str | None = None


class Address(BaseModel):
    """Structured postal address."""

    model_config = ConfigDict(extra="forbid")

    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    county: str | None = None


# ---------------------------------------------------------------------------
# Section A – Patient & demographics
# ---------------------------------------------------------------------------


class PatientInfo(BaseModel):
    """Patient demographics for NEMSIS ePatient and X12/FHIR Patient.

    ``sex_at_birth`` uses the NEMSIS / HL7 administrative-sex vocabulary:
        "M" = Male, "F" = Female, "U" = Unknown/undifferentiated.

    ``age_years`` is populated when DOB is unavailable (common in EMS audio
    where the crew states the age directly without a birth date).
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = None
    dob: date | None = None
    sex_at_birth: Literal["M", "F", "U"] | None = None
    gender_identity: str | None = None
    race: list[CodedValue] = Field(default_factory=list)
    ethnicity: CodedValue | None = None
    address: Address | None = None
    phone: str | None = None
    # Fallback fields populated from transcript when DOB is unavailable
    age_years: int | None = None
    age_hint_text: str | None = None  # raw text, e.g. "54 year old male"


# ---------------------------------------------------------------------------
# Section B – Subscriber / coverage / payer
# ---------------------------------------------------------------------------


class SubscriberInfo(BaseModel):
    """Insurance / payer details for X12 2000B/2000C loops and FHIR Coverage."""

    model_config = ConfigDict(extra="forbid")

    subscriber_name: str | None = None
    member_id: str | None = None
    group_id: str | None = None
    policy_id: str | None = None
    payer_id: str | None = None
    payer_name: str | None = None
    relationship_to_patient: str | None = None  # "self", "spouse", "child", etc.
    plan_type: str | None = None  # "Medicare", "Medicaid", "commercial", etc.


# ---------------------------------------------------------------------------
# Section C – Encounter / incident / admin
# ---------------------------------------------------------------------------


class EncounterInfo(BaseModel):
    """Operational encounter identifiers for NEMSIS eResponse and HL7 Encounter."""

    model_config = ConfigDict(extra="forbid")

    encounter_id: str | None = None
    run_number: str | None = None
    cad_incident_number: str | None = None
    pcr_number: str | None = None
    agency_id: str | None = None
    agency_name: str | None = None
    service_date: date | None = None
    service_start_datetime: datetime | None = None
    service_end_datetime: datetime | None = None


# ---------------------------------------------------------------------------
# Section D – Crew / unit / provider
# ---------------------------------------------------------------------------


class CrewMember(BaseModel):
    """Individual crew member.  ``certification_level`` follows NEMSIS eCrew codes."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    role: str | None = None                # "Paramedic", "EMT-Basic", "Driver"
    certification_level: str | None = None  # NEMSIS eCrew.14 codes
    npi: str | None = None
    state_license: str | None = None


class UnitInfo(BaseModel):
    """EMS unit (vehicle + crew) for NEMSIS eResponse and X12 rendering provider."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str | None = None
    vehicle_id: str | None = None
    agency_id: str | None = None
    crew: list[CrewMember] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section E – Timeline / events
# ---------------------------------------------------------------------------


class IncidentTimeline(BaseModel):
    """Operational timestamps for NEMSIS eTimes section.

    All times are ``datetime`` objects (timezone-naive UTC-assumed unless the
    source audio contained explicit timezone information).
    """

    model_config = ConfigDict(extra="forbid")

    psap_call_datetime: datetime | None = None
    dispatch_notified_datetime: datetime | None = None
    unit_enroute_datetime: datetime | None = None
    unit_on_scene_datetime: datetime | None = None
    patient_contact_datetime: datetime | None = None
    transport_begin_datetime: datetime | None = None
    arrive_destination_datetime: datetime | None = None
    transfer_of_care_datetime: datetime | None = None


# ---------------------------------------------------------------------------
# Section F – Clinical
# ---------------------------------------------------------------------------


class VitalSign(BaseModel):
    """A single vital sign observation for NEMSIS eVitals and FHIR Observation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str  # string to handle compound values like "120/80"
    unit: str | None = None
    timestamp: datetime | None = None
    method: str | None = None  # "auscultated", "automatic", "manual", etc.
    evidence_segment_ids: list[str] = Field(default_factory=list)


class Medication(BaseModel):
    """A medication administration for NEMSIS eMedications and FHIR MedicationAdministration."""

    model_config = ConfigDict(extra="forbid")

    drug: str
    dose: str | None = None
    route: str | None = None
    timestamp: datetime | None = None
    administered_by: str | None = None
    coded: CodedValue | None = None
    evidence_segment_ids: list[str] = Field(default_factory=list)


class Procedure(BaseModel):
    """A clinical procedure for NEMSIS eProcedures and FHIR Procedure."""

    model_config = ConfigDict(extra="forbid")

    name: str
    coded: CodedValue | None = None
    timestamp: datetime | None = None
    performed_by: str | None = None
    evidence_segment_ids: list[str] = Field(default_factory=list)


class ClinicalInfo(BaseModel):
    """Clinical findings section (NEMSIS eSituation / ePatientCare, FHIR Encounter)."""

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str | None = None
    primary_impression: CodedValue | None = None
    secondary_impressions: list[CodedValue] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    vitals: list[VitalSign] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    pmh: list[str] = Field(default_factory=list)  # past medical history


# ---------------------------------------------------------------------------
# Section G – Transport / disposition
# ---------------------------------------------------------------------------


class TransportInfo(BaseModel):
    """Transport and disposition details for NEMSIS eDisposition and X12 CLM."""

    model_config = ConfigDict(extra="forbid")

    origin_address: Address | None = None
    origin_facility_id: str | None = None
    destination_address: Address | None = None
    destination_facility_id: str | None = None
    destination_facility_name: str | None = None
    transport_mode: str | None = None       # "ALS", "BLS", "Air", etc.
    disposition_status: str | None = None   # "transported", "refused", "canceled", "unknown"
    disposition_code: CodedValue | None = None  # NEMSIS eDisposition.27 coded


# ---------------------------------------------------------------------------
# Section H – Billing / coding
# ---------------------------------------------------------------------------


class ServiceLine(BaseModel):
    """A single service line for X12 837 SV1 / FHIR Claim.item."""

    model_config = ConfigDict(extra="forbid")

    line_number: int
    procedure_code: str       # CPT or HCPCS
    modifiers: list[str] = Field(default_factory=list)
    diagnosis_pointers: list[int] = Field(default_factory=list)
    units: float = 1.0
    charge_amount: float | None = None
    description: str | None = None
    place_of_service: str | None = None   # CMS POS code (e.g. "41" = Ambulance Land)


class BillingInfo(BaseModel):
    """Billing and coding details for X12 837 and FHIR Claim."""

    model_config = ConfigDict(extra="forbid")

    claim_type: str | None = None           # "professional", "institutional"
    place_of_service: str | None = None     # default POS code for the encounter
    service_lines: list[ServiceLine] = Field(default_factory=list)
    icd10_diagnoses: list[CodedValue] = Field(default_factory=list)
    rendering_provider_npi: str | None = None
    billing_provider_npi: str | None = None
    billing_provider_name: str | None = None
    billing_provider_tax_id: str | None = None
    medical_necessity_notes: str | None = None


# ---------------------------------------------------------------------------
# Section J – Provenance / audit
# ---------------------------------------------------------------------------


class CanonicalProvenance(BaseModel):
    """Links a canonical field back to transcript evidence."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    entity_index: int | None = Field(default=None, ge=0)
    field_path: str | None = None   # dotted path, e.g. "patient.sex_at_birth"
    note: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


# ---------------------------------------------------------------------------
# Top-level canonical claim
# ---------------------------------------------------------------------------


class CanonicalClaim(BaseModel):
    """Version 1.0 canonical EMS claim with fully-structured sections.

    This is the interchange format consumed by the exporters (NEMSIS, X12 837,
    FHIR).  It is produced by ``enrich_canonical_claim()`` and can be serialised
    to JSON via ``model_dump(mode="json")``.

    Backward compatibility:
        The v0.1 ``Claim`` (``schema_version="0.1"``) remains unchanged in
        ``ems_pipeline.models``.  This model is additive and lives alongside it.
        The ``legacy_claim_id`` field carries the v0.1 ``claim_id`` for
        traceability.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    claim_id: str

    # Sections A–J
    patient: PatientInfo = Field(default_factory=PatientInfo)
    subscriber: SubscriberInfo = Field(default_factory=SubscriberInfo)
    encounter: EncounterInfo = Field(default_factory=EncounterInfo)
    unit: UnitInfo = Field(default_factory=UnitInfo)
    timeline: IncidentTimeline = Field(default_factory=IncidentTimeline)
    clinical: ClinicalInfo = Field(default_factory=ClinicalInfo)
    transport: TransportInfo = Field(default_factory=TransportInfo)
    billing: BillingInfo = Field(default_factory=BillingInfo)
    provenance: list[CanonicalProvenance] = Field(default_factory=list)

    # Migration traceability
    legacy_claim_id: str | None = None  # v0.1 claim_id for linkage


__all__ = [
    "Address",
    "BillingInfo",
    "CanonicalClaim",
    "CanonicalProvenance",
    "ClinicalInfo",
    "CodedValue",
    "CrewMember",
    "EncounterInfo",
    "IncidentTimeline",
    "Medication",
    "PatientInfo",
    "Procedure",
    "ServiceLine",
    "SubscriberInfo",
    "TransportInfo",
    "UnitInfo",
    "VitalSign",
]
