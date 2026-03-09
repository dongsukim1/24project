"""FHIR R4 Bundle exporter.

Maps a ``CanonicalClaim`` to a FHIR R4 Bundle (JSON dict) containing:

    Patient              — demographics (Section A)
    Coverage             — payer / subscriber (Section B)
    Encounter            — encounter admin + clinical context (Section C/F)
    Claim                — billing claim (Section H)
    Observation          — one per vital sign (Section F)
    Procedure            — one per clinical procedure (Section F)
    MedicationAdministration — one per medication (Section F)

This is an MVP mapping.  FHIR profiles (e.g. US Core, CARIN BlueButton) add
additional must-support elements; those are noted as warnings when absent.

FHIR R4 specification: https://hl7.org/fhir/R4/
"""

from __future__ import annotations

import uuid
from typing import Any

from ems_pipeline.claim.canonical import CanonicalClaim, VitalSign, Medication, Procedure
from ems_pipeline.exporters import ExportResult, fmt_dt, fmt_date, fmt_name

# ---------------------------------------------------------------------------
# Required FHIR fields (MVP subset)
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "Patient.name",
    "Encounter.status",
    "Encounter.class",
    "Claim.status",
    "Claim.use",
    "Claim.patient (reference)",
    "Claim.insurer (reference)",
]

_OPTIONAL_FIELDS = [
    "Patient.birthDate",
    "Patient.gender",
    "Coverage.subscriberId",
    "Observation.value[x] (vitals)",
    "Procedure.code",
    "MedicationAdministration.medication[x]",
]


# ---------------------------------------------------------------------------
# FHIR value maps
# ---------------------------------------------------------------------------

_SEX_MAP = {"M": "male", "F": "female", "U": "unknown"}

_VITAL_LOINC: dict[str, str] = {
    "bp": "55284-4",      # Blood pressure systolic/diastolic
    "blood pressure": "55284-4",
    "spo2": "59408-5",    # Oxygen saturation
    "oxygen saturation": "59408-5",
    "hr": "8867-4",       # Heart rate
    "heart rate": "8867-4",
    "pulse": "8867-4",
    "rr": "9279-1",       # Respiratory rate
    "respiratory rate": "9279-1",
    "temp": "8310-5",     # Body temperature
    "temperature": "8310-5",
    "gcs": "9269-2",      # Glasgow Coma Scale total
    "pain": "72514-3",    # Pain severity
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _ref(resource_type: str, rid: str) -> dict[str, str]:
    return {"reference": f"{resource_type}/{rid}"}


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def _build_patient(claim: CanonicalClaim, patient_id: str) -> dict[str, Any]:
    p = claim.patient
    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"]},
    }
    last, first = fmt_name(p.full_name)
    if p.full_name:
        resource["name"] = [{"use": "official", "family": last, "given": [first]}]
    else:
        resource["name"] = [{"use": "unknown", "text": "Unknown"}]

    if p.sex_at_birth:
        resource["gender"] = _SEX_MAP.get(p.sex_at_birth, "unknown")

    if p.dob:
        resource["birthDate"] = fmt_date(p.dob)
    elif p.age_years is not None:
        resource["_birthDate"] = {"extension": [{"url": "age-estimate", "valueInteger": p.age_years}]}

    if p.address:
        addr: dict[str, Any] = {"use": "home"}
        if p.address.street:
            addr["line"] = [p.address.street]
        if p.address.city:
            addr["city"] = p.address.city
        if p.address.state:
            addr["state"] = p.address.state
        if p.address.zip_code:
            addr["postalCode"] = p.address.zip_code
        resource["address"] = [addr]

    if p.phone:
        resource["telecom"] = [{"system": "phone", "value": p.phone}]

    return resource


def _build_coverage(claim: CanonicalClaim, coverage_id: str, patient_id: str) -> dict[str, Any]:
    s = claim.subscriber
    resource: dict[str, Any] = {
        "resourceType": "Coverage",
        "id": coverage_id,
        "status": "active",
        "beneficiary": _ref("Patient", patient_id),
    }
    if s.payer_id or s.payer_name:
        resource["payor"] = [{"identifier": {"value": s.payer_id}, "display": s.payer_name}]
    if s.member_id:
        resource["subscriberId"] = s.member_id
    if s.group_id:
        resource["class"] = [{"type": {"coding": [{"code": "group"}]}, "value": s.group_id}]
    if s.plan_type:
        resource["type"] = {"coding": [{"code": s.plan_type}]}
    return resource


def _build_encounter(
    claim: CanonicalClaim, encounter_id: str, patient_id: str
) -> dict[str, Any]:
    t = claim.timeline
    enc = claim.encounter
    period: dict[str, Any] = {}
    if t.unit_on_scene_datetime:
        period["start"] = fmt_dt(t.unit_on_scene_datetime)
    elif t.patient_contact_datetime:
        period["start"] = fmt_dt(t.patient_contact_datetime)
    if t.transfer_of_care_datetime:
        period["end"] = fmt_dt(t.transfer_of_care_datetime)
    elif t.arrive_destination_datetime:
        period["end"] = fmt_dt(t.arrive_destination_datetime)

    resource: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "EMER",
            "display": "emergency",
        },
        "subject": _ref("Patient", patient_id),
        "serviceType": {
            "coding": [{"system": "http://snomed.info/sct", "code": "409971007", "display": "Emergency medical services"}]
        },
    }
    if period:
        resource["period"] = period

    # Reason (chief complaint / primary impression)
    c = claim.clinical
    if c.chief_complaint:
        resource["reasonCode"] = [{"text": c.chief_complaint}]
    if c.primary_impression and c.primary_impression.code != "UNKNOWN":
        resource["diagnosis"] = [
            {
                "condition": {"display": c.primary_impression.display or c.primary_impression.code},
                "rank": 1,
            }
        ]
    if enc.run_number:
        resource["identifier"] = [
            {"system": "urn:ems:run_number", "value": enc.run_number}
        ]
    return resource


def _build_claim_resource(
    claim: CanonicalClaim,
    fhir_claim_id: str,
    patient_id: str,
    coverage_id: str,
    encounter_id: str,
) -> dict[str, Any]:
    b = claim.billing
    enc = claim.encounter
    svc_date = fmt_date(enc.service_date or enc.service_start_datetime)

    resource: dict[str, Any] = {
        "resourceType": "Claim",
        "id": fhir_claim_id,
        "status": "active",
        "use": "claim",
        "patient": _ref("Patient", patient_id),
        "created": svc_date or "unknown",
        "insurer": {"identifier": {"value": claim.subscriber.payer_id or "UNKNOWN"}},
        "provider": {"identifier": {"value": b.billing_provider_npi or "UNKNOWN"}},
        "priority": {"coding": [{"code": "normal"}]},
        "insurance": [
            {
                "sequence": 1,
                "focal": True,
                "coverage": _ref("Coverage", coverage_id),
            }
        ],
        "encounter": [_ref("Encounter", encounter_id)],
    }

    if b.icd10_diagnoses:
        resource["diagnosis"] = [
            {
                "sequence": idx + 1,
                "diagnosisCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": dx.code,
                            "display": dx.display,
                        }
                    ]
                },
            }
            for idx, dx in enumerate(b.icd10_diagnoses)
        ]

    if b.service_lines:
        resource["item"] = [
            {
                "sequence": sl.line_number,
                "productOrService": {
                    "coding": [
                        {
                            "system": "http://www.ama-assn.org/go/cpt",
                            "code": sl.procedure_code,
                            "display": sl.description,
                        }
                    ]
                },
                "quantity": {"value": sl.units},
                "unitPrice": {"value": sl.charge_amount, "currency": "USD"}
                if sl.charge_amount
                else None,
                "informationSequence": sl.diagnosis_pointers,
            }
            for sl in b.service_lines
        ]

    if b.medical_necessity_notes:
        resource["supportingInfo"] = [
            {
                "sequence": 1,
                "category": {"coding": [{"code": "info"}]},
                "valueString": b.medical_necessity_notes,
            }
        ]

    return resource


def _loinc_for_vital(name: str) -> str | None:
    return _VITAL_LOINC.get(name.lower().strip())


def _build_observation(
    vital: VitalSign, obs_id: str, patient_id: str, encounter_id: str
) -> dict[str, Any]:
    loinc = _loinc_for_vital(vital.name)
    coding: list[dict[str, Any]] = []
    if loinc:
        coding.append(
            {"system": "http://loinc.org", "code": loinc, "display": vital.name}
        )
    coding.append({"system": "urn:ems:vital", "code": vital.name})

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                    }
                ]
            }
        ],
        "code": {"coding": coding, "text": vital.name},
        "subject": _ref("Patient", patient_id),
        "encounter": _ref("Encounter", encounter_id),
        "valueString": vital.value,
    }
    if vital.unit:
        resource["valueQuantity"] = {
            "value": vital.value,
            "unit": vital.unit,
            "system": "http://unitsofmeasure.org",
        }
    if vital.timestamp:
        resource["effectiveDateTime"] = fmt_dt(vital.timestamp)
    return resource


def _build_procedure(
    proc: Procedure, proc_id: str, patient_id: str, encounter_id: str
) -> dict[str, Any]:
    coding: list[dict[str, Any]] = []
    if proc.coded:
        coding.append(
            {
                "system": f"urn:{proc.coded.code_system.lower()}",
                "code": proc.coded.code,
                "display": proc.coded.display or proc.name,
            }
        )
    coding.append({"system": "urn:ems:procedure", "code": proc.name})

    resource: dict[str, Any] = {
        "resourceType": "Procedure",
        "id": proc_id,
        "status": "completed",
        "code": {"coding": coding, "text": proc.name},
        "subject": _ref("Patient", patient_id),
        "encounter": _ref("Encounter", encounter_id),
    }
    if proc.timestamp:
        resource["performedDateTime"] = fmt_dt(proc.timestamp)
    return resource


def _build_medication_admin(
    med: Medication, med_id: str, patient_id: str, encounter_id: str
) -> dict[str, Any]:
    coding: list[dict[str, Any]] = []
    if med.coded:
        coding.append(
            {
                "system": f"urn:{med.coded.code_system.lower()}",
                "code": med.coded.code,
                "display": med.coded.display or med.drug,
            }
        )
    coding.append({"system": "urn:ems:medication", "code": med.drug})

    resource: dict[str, Any] = {
        "resourceType": "MedicationAdministration",
        "id": med_id,
        "status": "completed",
        "medicationCodeableConcept": {"coding": coding, "text": med.drug},
        "subject": _ref("Patient", patient_id),
        "context": _ref("Encounter", encounter_id),
        "effectiveDateTime": fmt_dt(med.timestamp) or "unknown",
    }
    dosage: dict[str, Any] = {}
    if med.dose:
        dosage["text"] = med.dose
    if med.route:
        dosage["route"] = {"text": med.route}
    if dosage:
        resource["dosage"] = dosage
    return resource


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_required(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    if not claim.patient.full_name:
        missing.append("patient.full_name (Patient.name)")
    if not claim.subscriber.payer_id and not claim.subscriber.payer_name:
        missing.append("subscriber.payer_id or payer_name (Claim.insurer)")
    return missing


def _check_optional(claim: CanonicalClaim) -> list[str]:
    missing: list[str] = []
    if claim.patient.dob is None and claim.patient.age_years is None:
        missing.append("patient.dob (Patient.birthDate)")
    if claim.patient.sex_at_birth is None:
        missing.append("patient.sex_at_birth (Patient.gender)")
    if not claim.subscriber.member_id:
        missing.append("subscriber.member_id (Coverage.subscriberId)")
    if not claim.clinical.vitals:
        missing.append("clinical.vitals (Observation resources)")
    if not claim.clinical.procedures:
        missing.append("clinical.procedures (Procedure resources)")
    if not claim.clinical.medications:
        missing.append("clinical.medications (MedicationAdministration resources)")
    return missing


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def export_fhir(claim: CanonicalClaim) -> ExportResult:
    """Export a ``CanonicalClaim`` to a FHIR R4 Bundle dict.

    Args:
        claim: The v1.0 canonical claim to export.

    Returns:
        An ``ExportResult`` whose ``data`` is a FHIR R4 Bundle (JSON-serialisable
        dict) plus validation diagnostics.
    """
    patient_id = _new_id()
    coverage_id = _new_id()
    encounter_id = claim.encounter.encounter_id or _new_id()
    fhir_claim_id = _new_id()

    entries: list[dict[str, Any]] = []

    patient_res = _build_patient(claim, patient_id)
    entries.append({"fullUrl": f"urn:uuid:{patient_id}", "resource": patient_res})

    coverage_res = _build_coverage(claim, coverage_id, patient_id)
    entries.append({"fullUrl": f"urn:uuid:{coverage_id}", "resource": coverage_res})

    encounter_res = _build_encounter(claim, encounter_id, patient_id)
    entries.append({"fullUrl": f"urn:uuid:{encounter_id}", "resource": encounter_res})

    claim_res = _build_claim_resource(
        claim, fhir_claim_id, patient_id, coverage_id, encounter_id
    )
    entries.append({"fullUrl": f"urn:uuid:{fhir_claim_id}", "resource": claim_res})

    for vital in claim.clinical.vitals:
        obs_id = _new_id()
        entries.append(
            {
                "fullUrl": f"urn:uuid:{obs_id}",
                "resource": _build_observation(vital, obs_id, patient_id, encounter_id),
            }
        )

    for proc in claim.clinical.procedures:
        proc_id = _new_id()
        entries.append(
            {
                "fullUrl": f"urn:uuid:{proc_id}",
                "resource": _build_procedure(proc, proc_id, patient_id, encounter_id),
            }
        )

    for med in claim.clinical.medications:
        med_id = _new_id()
        entries.append(
            {
                "fullUrl": f"urn:uuid:{med_id}",
                "resource": _build_medication_admin(
                    med, med_id, patient_id, encounter_id
                ),
            }
        )

    data: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": _new_id(),
        "type": "collection",
        "entry": entries,
        "_meta": {
            "source_claim_id": claim.claim_id,
            "schema_version": claim.schema_version,
            "fhir_version": "R4",
        },
    }

    warnings: list[str] = []
    if claim.clinical.primary_impression and claim.clinical.primary_impression.code == "UNKNOWN":
        warnings.append(
            "Encounter.diagnosis code is placeholder 'UNKNOWN' — "
            "requires ICD-10-CM mapping before submission"
        )
    if not claim.billing.icd10_diagnoses:
        warnings.append(
            "Claim.diagnosis is empty — at least one ICD-10-CM code required for billing"
        )

    return ExportResult(
        format="fhir",
        data=data,
        missing_required=_check_required(claim),
        missing_optional=_check_optional(claim),
        warnings=warnings,
    )
