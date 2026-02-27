"""Enrichment layer: extracted entities + transcript → CanonicalClaim (v1.0).

This module is deterministic (rules-based, no ML).  It reads ``Entity`` objects
produced by the extraction stage and maps them to the canonical strongly-typed
sections defined in ``canonical.py``.

For fields that cannot be populated from available data a ``None`` value (or
empty list/object) is used as the explicit placeholder.  A ``CanonicalProvenance``
entry is added for every populated field so downstream tools can trace back to
transcript evidence.

Usage::

    from ems_pipeline.claim.enrich import enrich_canonical_claim
    from ems_pipeline.claim.timeline import build_events

    events = build_events(transcript, entities)
    canonical = enrich_canonical_claim(
        claim_id=legacy_claim.claim_id,
        transcript=transcript,
        entities=entities,
        events=events,
        legacy_claim=legacy_claim,
        agent2_codes=session.code_suggestions,
        payer_id=session.payer_id,
    )
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from ems_pipeline.claim.builder import (
    _all_entities,
    _evidence_for_entity,
    _first_entity,
    _segment_ids,
    _unique_strs,
)
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
from ems_pipeline.models import Claim, Entity, Transcript

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_SEX_NORM: dict[str, str] = {
    "male": "M",
    "m": "M",
    "man": "M",
    "boy": "M",
    "female": "F",
    "f": "F",
    "woman": "F",
    "girl": "F",
    "unknown": "U",
    "u": "U",
}

_AGE_RE = re.compile(r"(\d+)\s*(?:year|yr|y\.?o\.?|y/o)", re.IGNORECASE)

# ICD-10-CM: requires a decimal point OR is 3 chars (category only, rare as standalone)
# Full ICD-10 codes look like "R07.9", "I21.9", "J18.0", "Z87.891"
_ICD10_FULL_RE = re.compile(r"^[A-Z][0-9]{2}\.[0-9A-Z]{1,4}$", re.IGNORECASE)
# HCPCS Level II: exactly 5 chars — one letter then 4 digits (e.g. A0427, G0180)
_HCPCS_RE = re.compile(r"^[A-Z][0-9]{4}$", re.IGNORECASE)
# CPT: exactly 5 digits (e.g. 99283) or 4 digits + category-I modifier letter
_CPT_RE = re.compile(r"^[0-9]{4,5}[A-Z0-9]?$", re.IGNORECASE)


def _try_parse_age(text: str) -> int | None:
    """Extract a numeric age from free text like '54 year old' or '54'."""
    m = _AGE_RE.search(text)
    if m:
        return int(m.group(1))
    try:
        v = int(text.strip())
        return v if 0 < v < 130 else None
    except ValueError:
        return None


def _prov(
    segment_id: str,
    entity_index: int | None,
    field_path: str,
    confidence: float | None,
) -> CanonicalProvenance:
    return CanonicalProvenance(
        segment_id=segment_id,
        entity_index=entity_index,
        field_path=field_path,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Section A – Patient
# ---------------------------------------------------------------------------


def _map_patient(
    transcript: Transcript,
    entities: Sequence[Entity],
) -> tuple[PatientInfo, list[CanonicalProvenance]]:
    provenance: list[CanonicalProvenance] = []

    age_result = _first_entity(entities, {"age", "patient_age"})
    sex_result = _first_entity(entities, {"sex", "gender", "patient_sex"})
    name_result = _first_entity(entities, {"patient_name", "name"})

    age_years: int | None = None
    age_hint: str | None = None
    if age_result is not None:
        i, ent = age_result
        age_hint = ent.normalized or ent.text
        if age_hint:
            age_years = _try_parse_age(age_hint)
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "patient.age_years", ent.confidence))

    sex_at_birth: str | None = None
    if sex_result is not None:
        i, ent = sex_result
        raw = (ent.normalized or ent.text or "").strip().lower()
        sex_at_birth = _SEX_NORM.get(raw)
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "patient.sex_at_birth", ent.confidence))

    full_name: str | None = None
    if name_result is not None:
        i, ent = name_result
        full_name = ent.normalized or ent.text
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "patient.full_name", ent.confidence))

    patient = PatientInfo(
        full_name=full_name,
        sex_at_birth=sex_at_birth,  # type: ignore[arg-type]
        age_years=age_years,
        age_hint_text=age_hint,
    )
    return patient, provenance


# ---------------------------------------------------------------------------
# Section F – Clinical
# ---------------------------------------------------------------------------


def _map_clinical(
    transcript: Transcript,
    entities: Sequence[Entity],
) -> tuple[ClinicalInfo, list[CanonicalProvenance]]:
    provenance: list[CanonicalProvenance] = []

    # Chief complaint / primary impression
    cc_result = _first_entity(
        entities, {"chief_complaint", "complaint", "primary_impression"}
    )
    if cc_result is None:
        cc_result = _first_entity(entities, {"symptom"})

    chief_complaint: str | None = None
    primary_impression: CodedValue | None = None
    if cc_result is not None:
        i, ent = cc_result
        chief_complaint = ent.normalized or ent.text
        if chief_complaint:
            primary_impression = CodedValue(
                code="UNKNOWN",
                code_system="ICD-10-CM",
                display=chief_complaint,
                confidence=ent.confidence,
                source="extracted",
            )
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "clinical.chief_complaint", ent.confidence))

    # Symptoms
    symptoms: list[str] = []
    for i, ent in _all_entities(entities, {"symptom"}):
        text = ent.normalized or ent.text
        if text:
            symptoms.append(text)
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "clinical.symptoms", ent.confidence))

    # Vitals
    vitals: list[VitalSign] = []
    for i, ent in _all_entities(
        entities, {"vital", "vitals", "vital_bp", "vital_spo2"}
    ):
        attrs: dict[str, Any] = (
            ent.attributes if isinstance(ent.attributes, dict) else {}
        )
        ent_type = (ent.type or "").strip().lower()
        name: str = attrs.get("name") or ent.normalized or ent.text or "UNKNOWN"
        value: str = str(attrs.get("value") or ent.normalized or ent.text or "UNKNOWN")
        unit: str | None = (
            str(attrs.get("unit")) if attrs.get("unit") is not None else None
        )
        # Synthesise unit for typed vital sub-types when absent
        if not unit and ent_type == "vital_bp":
            unit = "mmHg"
        if not unit and ent_type == "vital_spo2":
            unit = "%"
        ev_ids = _evidence_for_entity(transcript, ent)
        vitals.append(VitalSign(name=str(name), value=value, unit=unit, evidence_segment_ids=ev_ids))
        for s in ev_ids:
            provenance.append(_prov(s, i, "clinical.vitals", ent.confidence))

    # Medications
    medications: list[Medication] = []
    for i, ent in _all_entities(entities, {"medication"}):
        drug = ent.normalized or ent.text or "UNKNOWN"
        attrs = ent.attributes if isinstance(ent.attributes, dict) else {}
        dose = attrs.get("dose")
        route = attrs.get("route")
        ev_ids = _evidence_for_entity(transcript, ent)
        medications.append(
            Medication(
                drug=drug,
                dose=str(dose) if dose is not None else None,
                route=str(route) if route is not None else None,
                evidence_segment_ids=ev_ids,
            )
        )
        for s in ev_ids:
            provenance.append(_prov(s, i, "clinical.medications", ent.confidence))

    # Procedures
    procedures: list[Procedure] = []
    for i, ent in _all_entities(entities, {"procedure", "intervention", "treatment"}):
        name = ent.normalized or ent.text or "UNKNOWN"
        ev_ids = _evidence_for_entity(transcript, ent)
        procedures.append(Procedure(name=name, evidence_segment_ids=ev_ids))
        for s in ev_ids:
            provenance.append(_prov(s, i, "clinical.procedures", ent.confidence))

    clinical = ClinicalInfo(
        chief_complaint=chief_complaint,
        primary_impression=primary_impression,
        symptoms=symptoms,
        vitals=vitals,
        medications=medications,
        procedures=procedures,
    )
    return clinical, provenance


# ---------------------------------------------------------------------------
# Section G – Transport
# ---------------------------------------------------------------------------


def _map_transport(
    transcript: Transcript,
    entities: Sequence[Entity],
    events: Sequence[dict[str, Any]],
) -> tuple[TransportInfo, list[CanonicalProvenance]]:
    provenance: list[CanonicalProvenance] = []

    disp_result = _first_entity(entities, {"disposition"})
    dest_result = _first_entity(
        entities, {"destination", "destination_hint", "hospital"}
    )
    origin_result = _first_entity(entities, {"location", "address", "cross_street"})

    disposition_status: str | None = None
    if disp_result is not None:
        i, ent = disp_result
        norm = (ent.normalized or ent.text or "").strip().lower()
        if any(k in norm for k in ("refus", "ama")):
            disposition_status = "refused"
        elif any(k in norm for k in ("cancel",)):
            disposition_status = "canceled"
        elif any(k in norm for k in ("transport",)):
            disposition_status = "transported"
        else:
            disposition_status = norm or "unknown"
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "transport.disposition_status", ent.confidence))
    elif any(e.get("type") in ("TRANSPORT_BEGIN", "ARRIVE_ED") for e in events):
        disposition_status = "transported"

    destination_name: str | None = None
    if dest_result is not None:
        i, ent = dest_result
        destination_name = ent.normalized or ent.text
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "transport.destination_facility_name", ent.confidence))

    origin_address: Address | None = None
    if origin_result is not None:
        i, ent = origin_result
        text = ent.normalized or ent.text
        if text:
            origin_address = Address(street=text)
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "transport.origin_address", ent.confidence))

    transport = TransportInfo(
        origin_address=origin_address,
        destination_facility_name=destination_name,
        disposition_status=disposition_status or "unknown",
    )
    return transport, provenance


# ---------------------------------------------------------------------------
# Section D – Unit
# ---------------------------------------------------------------------------


def _map_unit(
    transcript: Transcript,
    entities: Sequence[Entity],
) -> tuple[UnitInfo, list[CanonicalProvenance]]:
    provenance: list[CanonicalProvenance] = []

    unit_results = _all_entities(entities, {"unit", "unit_id"})
    crew_results = _all_entities(entities, {"crew", "crew_member", "provider"})

    unit_id: str | None = None
    if unit_results:
        i, ent = unit_results[0]
        unit_id = ent.normalized or ent.text
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "unit.unit_id", ent.confidence))

    crew: list[CrewMember] = []
    for i, ent in crew_results:
        name = ent.normalized or ent.text
        crew.append(CrewMember(name=name))
        for s in _evidence_for_entity(transcript, ent):
            provenance.append(_prov(s, i, "unit.crew", ent.confidence))

    return UnitInfo(unit_id=unit_id, crew=crew), provenance


# ---------------------------------------------------------------------------
# Section C – Encounter
# ---------------------------------------------------------------------------


def _map_encounter(
    transcript: Transcript,
    claim_id: str,
) -> EncounterInfo:
    """Populate encounter fields from transcript metadata and claim_id."""
    md: dict[str, Any] = (
        transcript.metadata if isinstance(transcript.metadata, dict) else {}
    )

    def _str_or_none(key: str) -> str | None:
        v = md.get(key)
        return str(v) if v is not None else None

    return EncounterInfo(
        encounter_id=claim_id,
        run_number=_str_or_none("run_number") or _str_or_none("run_id"),
        cad_incident_number=_str_or_none("cad_incident_number")
        or _str_or_none("cad_number"),
        pcr_number=_str_or_none("pcr_number") or _str_or_none("pcr_id"),
        agency_id=_str_or_none("agency_id"),
        agency_name=_str_or_none("agency_name"),
    )


# ---------------------------------------------------------------------------
# Section H – Billing
# ---------------------------------------------------------------------------


def _classify_code(code: str) -> str:
    """Return 'icd10', 'cpt', or 'unknown' for a bare code string.

    Priority:
    1. Decimal-containing alpha codes → ICD-10-CM (e.g. "R07.9", "I21.9").
    2. HCPCS Level II pattern (letter + 4 digits, no decimal) → CPT/HCPCS
       service code (e.g. "A0427", "G0180").
    3. Pure-digit CPT codes (5 digits) → CPT.
    4. Everything else → unknown.
    """
    c = code.strip().upper()
    if _ICD10_FULL_RE.match(c):
        return "icd10"
    if _HCPCS_RE.match(c) or _CPT_RE.match(c):
        return "cpt"
    return "unknown"


def _map_billing(
    agent2_codes: list[str] | None,
) -> BillingInfo:
    """Build a minimal billing section from Agent 2 code suggestions."""
    service_lines: list[ServiceLine] = []
    icd10: list[CodedValue] = []

    for code in agent2_codes or []:
        code = code.strip()
        if not code:
            continue
        kind = _classify_code(code)
        if kind == "icd10":
            icd10.append(
                CodedValue(code=code, code_system="ICD-10-CM", source="agent2")
            )
        else:
            # Treat as CPT/HCPCS service line
            service_lines.append(
                ServiceLine(
                    line_number=len(service_lines) + 1,
                    procedure_code=code,
                    place_of_service="41",  # Ambulance Land
                    diagnosis_pointers=[1] if icd10 else [],
                )
            )

    return BillingInfo(
        claim_type="professional",
        place_of_service="41",
        service_lines=service_lines,
        icd10_diagnoses=icd10,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_canonical_claim(
    claim_id: str,
    transcript: Transcript,
    entities: Sequence[Entity],
    events: Sequence[dict[str, Any]],
    legacy_claim: Claim | None = None,
    agent2_codes: list[str] | None = None,
    payer_id: str | None = None,
) -> CanonicalClaim:
    """Build a ``CanonicalClaim`` (v1.0) from pipeline outputs.

    This function is the single entry-point for enrichment.  It is intentionally
    deterministic: given the same inputs it always produces the same output.

    Args:
        claim_id:      Stable encounter identifier (usually from legacy claim hash
                       or a UUID assigned by the orchestrator).
        transcript:    Diarized transcript produced by Agent 1.
        entities:      Extracted entity list produced by Agent 1.
        events:        Timeline events produced by ``build_events()``.
        legacy_claim:  Optional v0.1 ``Claim`` to carry ``legacy_claim_id``
                       traceability.
        agent2_codes:  Optional list of CPT/ICD code strings from Agent 2.
        payer_id:      Optional payer identifier (e.g. from Agent 3 options).

    Returns:
        A fully-populated ``CanonicalClaim`` with ``None`` / empty-list
        placeholders for fields that could not be derived from the available data.
    """
    all_provenance: list[CanonicalProvenance] = []

    patient, prov = _map_patient(transcript, entities)
    all_provenance.extend(prov)

    clinical, prov = _map_clinical(transcript, entities)
    all_provenance.extend(prov)

    transport, prov = _map_transport(transcript, entities, events)
    all_provenance.extend(prov)

    unit, prov = _map_unit(transcript, entities)
    all_provenance.extend(prov)

    encounter = _map_encounter(transcript, claim_id)

    subscriber = SubscriberInfo(payer_id=payer_id)

    billing = _map_billing(agent2_codes)

    return CanonicalClaim(
        claim_id=claim_id,
        patient=patient,
        subscriber=subscriber,
        encounter=encounter,
        unit=unit,
        clinical=clinical,
        transport=transport,
        billing=billing,
        provenance=all_provenance,
        legacy_claim_id=legacy_claim.claim_id if legacy_claim else None,
    )
