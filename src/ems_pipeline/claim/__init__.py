"""Claim-building subpackage.

`ems_pipeline.claim` remains the stable import path for claim-building utilities.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from ems_pipeline.models import Claim, EntitiesDocument, Entity, ProvenanceLink


def build_claim(entities_doc: EntitiesDocument) -> Claim:
    """Build a structured proto-claim JSON from extracted entities.

    This is the CLI-facing claim builder. It intentionally consumes only the
    `EntitiesDocument` produced by `ems_pipeline extract`, so evidence linking is
    based on the per-entity `attributes.segment_id` emitted by the extractor.

    Args:
        entities_doc: Output from `ems_pipeline extract`.

    Returns:
        Claim: structured fields plus provenance links to segment IDs.
    """

    entities: Sequence[Entity] = entities_doc.entities or []

    def unique_strs(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            v = (v or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def entity_segment_ids(ent: Entity) -> list[str]:
        attrs = ent.attributes if isinstance(ent.attributes, dict) else {}
        seg_id = attrs.get("segment_id")
        if isinstance(seg_id, str) and seg_id.strip():
            return [seg_id.strip()]
        return []

    def value_with_evidence(
        value: Any,
        *,
        evidence_segment_ids: Iterable[str],
        evidence_entity_indexes: Iterable[int] = (),
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": value,
            "evidence_segment_ids": unique_strs(evidence_segment_ids),
        }
        idxs = sorted({int(i) for i in evidence_entity_indexes})
        if idxs:
            payload["evidence_entity_indexes"] = idxs
        return payload

    def first_entity(types: set[str]) -> tuple[int, Entity] | None:
        for i, ent in enumerate(entities):
            if (ent.type or "").strip().lower() in types:
                return i, ent
        return None

    def all_entities(types: set[str]) -> list[tuple[int, Entity]]:
        out: list[tuple[int, Entity]] = []
        for i, ent in enumerate(entities):
            if (ent.type or "").strip().lower() in types:
                out.append((i, ent))
        return out

    md = entities_doc.metadata if isinstance(entities_doc.metadata, dict) else {}
    transcript_norm = md.get("transcript_normalized")
    transcript_orig = md.get("transcript_original")
    if isinstance(transcript_norm, str) and transcript_norm.strip():
        claim_key = f"transcript_normalized:{transcript_norm.strip()}"
    elif isinstance(transcript_orig, str) and transcript_orig.strip():
        claim_key = f"transcript_original:{transcript_orig.strip()}"
    else:
        parts: list[str] = []
        for e in entities:
            parts.append(f"{(e.type or '').strip()}:{(e.normalized or e.text or '').strip()}")
        claim_key = "entities:" + "|".join(parts)
    claim_id = hashlib.sha256(claim_key.encode("utf-8")).hexdigest()

    provenance: list[ProvenanceLink] = []

    def add_provenance(
        *,
        seg_ids: Iterable[str],
        note: str,
        entity_index: int | None = None,
    ) -> None:
        for seg_id in unique_strs(seg_ids):
            conf: float | None = None
            if entity_index is not None and 0 <= entity_index < len(entities):
                conf = entities[entity_index].confidence
            provenance.append(
                ProvenanceLink(
                    segment_id=seg_id,
                    entity_index=entity_index,
                    note=note,
                    confidence=conf,
                )
            )

    fields: dict[str, Any] = {}

    # location_hint
    loc = first_entity({"location", "address", "cross_street"})
    if loc is not None:
        i, ent = loc
        seg_ids = entity_segment_ids(ent)
        fields["location_hint"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="location_hint", entity_index=i)
    else:
        fields["location_hint"] = value_with_evidence("", evidence_segment_ids=[])

    # dispatch
    dispatch: dict[str, Any] = {}
    dispatch_evidence: list[str] = []

    inc = first_entity({"incident_type", "call_type", "dispatch_reason", "condition"})
    if inc is not None:
        i, ent = inc
        seg_ids = entity_segment_ids(ent)
        dispatch_evidence.extend(seg_ids)
        dispatch["incident_type"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="dispatch.incident_type", entity_index=i)
    else:
        dispatch["incident_type"] = value_with_evidence("", evidence_segment_ids=dispatch_evidence)

    pr = first_entity({"priority", "acuity"})
    if pr is not None:
        i, ent = pr
        seg_ids = entity_segment_ids(ent)
        dispatch_evidence.extend(seg_ids)
        dispatch["priority"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="dispatch.priority", entity_index=i)
    else:
        dispatch["priority"] = value_with_evidence("", evidence_segment_ids=dispatch_evidence)

    units: list[dict[str, Any]] = []
    for i, ent in all_entities({"unit", "unit_id"}):
        seg_ids = entity_segment_ids(ent)
        dispatch_evidence.extend(seg_ids)
        units.append(
            value_with_evidence(
                ent.normalized or ent.text,
                evidence_segment_ids=seg_ids,
                evidence_entity_indexes=[i],
            )
        )
        add_provenance(seg_ids=seg_ids, note="dispatch.units", entity_index=i)
    dispatch["units"] = units
    dispatch["evidence_segment_ids"] = unique_strs(dispatch_evidence)
    fields["dispatch"] = dispatch

    # patient (optional)
    patient: dict[str, Any] = {}
    age = first_entity({"age", "patient_age"})
    sex = first_entity({"sex", "gender", "patient_sex"})
    if age is not None:
        i, ent = age
        seg_ids = entity_segment_ids(ent)
        patient["age_hint"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="patient.age_hint", entity_index=i)
    if sex is not None:
        i, ent = sex
        seg_ids = entity_segment_ids(ent)
        patient["sex_hint"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="patient.sex_hint", entity_index=i)
    if patient:
        fields["patient"] = patient

    # primary_impression / chief_complaint
    cc = first_entity({"chief_complaint", "complaint", "primary_impression"})
    if cc is None:
        cc = first_entity({"condition", "symptom"})
    if cc is not None:
        i, ent = cc
        seg_ids = entity_segment_ids(ent)
        fields["primary_impression"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="primary_impression", entity_index=i)
    else:
        fields["primary_impression"] = value_with_evidence("", evidence_segment_ids=[])

    # findings: symptoms + vitals (+ assessments)
    findings: dict[str, Any] = {}

    symptoms: list[dict[str, Any]] = []
    for i, ent in all_entities({"symptom"}):
        seg_ids = entity_segment_ids(ent)
        symptoms.append(
            value_with_evidence(
                ent.normalized or ent.text,
                evidence_segment_ids=seg_ids,
                evidence_entity_indexes=[i],
            )
        )
        add_provenance(seg_ids=seg_ids, note="findings.symptoms", entity_index=i)
    findings["symptoms"] = symptoms

    vitals: list[dict[str, Any]] = []
    for i, ent in all_entities({"vital", "vitals", "vital_bp", "vital_spo2"}):
        seg_ids = entity_segment_ids(ent)
        ent_type = (ent.type or "").strip().lower()
        name: str | None = None
        value: Any | None = None
        unit: str | None = None

        if isinstance(ent.attributes, dict):
            raw_name = ent.attributes.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
            raw_value = ent.attributes.get("value")
            value = raw_value if raw_value not in ("", None) else None
            raw_unit = ent.attributes.get("unit")
            if isinstance(raw_unit, str) and raw_unit.strip():
                unit = raw_unit.strip()

        if name is None and ent_type == "vital_bp":
            name = "BP"
            value = ent.normalized or ent.text
            unit = unit or "mmHg"
        if name is None and ent_type == "vital_spo2":
            name = "SpO2"
            value = ent.normalized or ent.text
            unit = unit or "%"

        if name is None:
            name = ent.normalized or ent.text
        if value is None:
            value = ent.normalized or ent.text

        vital_item: dict[str, Any] = {
            "name": name,
            "value": value,
            "unit": unit,
            "evidence_segment_ids": unique_strs(seg_ids),
            "evidence_entity_indexes": [i],
        }
        vitals.append({k: v for k, v in vital_item.items() if v not in (None, "", [])})
        add_provenance(seg_ids=seg_ids, note="findings.vitals", entity_index=i)
    findings["vitals"] = vitals

    assessments: list[dict[str, Any]] = []
    for i, ent in all_entities({"assessment"}):
        seg_ids = entity_segment_ids(ent)
        assessments.append(
            value_with_evidence(
                ent.normalized or ent.text,
                evidence_segment_ids=seg_ids,
                evidence_entity_indexes=[i],
            )
        )
        add_provenance(seg_ids=seg_ids, note="findings.assessments", entity_index=i)
    if assessments:
        findings["assessments"] = assessments

    fields["findings"] = findings

    # interventions: meds/procedures/resources
    interventions: list[dict[str, Any]] = []
    for i, ent in all_entities(
        {"medication", "procedure", "intervention", "treatment", "resource"}
    ):
        seg_ids = entity_segment_ids(ent)
        interventions.append(
            {
                "kind": (ent.type or "").strip().lower(),
                "text": ent.text,
                "normalized": ent.normalized,
                "evidence_segment_ids": unique_strs(seg_ids),
                "evidence_entity_indexes": [i],
            }
        )
        add_provenance(seg_ids=seg_ids, note="interventions", entity_index=i)
    fields["interventions"] = interventions

    # ETA (optional convenience field for the MVP extractor)
    eta = first_entity({"eta"})
    if eta is not None:
        i, ent = eta
        seg_ids = entity_segment_ids(ent)
        fields["eta"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="eta", entity_index=i)

    # disposition (best-effort, usually empty for the current extractor)
    disposition: dict[str, Any] = {}
    status_ent = first_entity({"disposition"})
    if status_ent is not None:
        i, ent = status_ent
        seg_ids = entity_segment_ids(ent)
        disposition["status"] = value_with_evidence(
            (ent.normalized or ent.text or "").strip().lower(),
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="disposition.status", entity_index=i)
    else:
        disposition["status"] = value_with_evidence("unknown", evidence_segment_ids=[])

    dest_ent = first_entity({"destination", "destination_hint", "hospital"})
    if dest_ent is not None:
        i, ent = dest_ent
        seg_ids = entity_segment_ids(ent)
        disposition["destination_hint"] = value_with_evidence(
            ent.normalized or ent.text,
            evidence_segment_ids=seg_ids,
            evidence_entity_indexes=[i],
        )
        add_provenance(seg_ids=seg_ids, note="disposition.destination_hint", entity_index=i)
    else:
        disposition["destination_hint"] = value_with_evidence("", evidence_segment_ids=[])

    disposition["evidence_segment_ids"] = unique_strs(
        [
            *disposition["status"].get("evidence_segment_ids", []),
            *disposition["destination_hint"].get("evidence_segment_ids", []),
        ]
    )
    fields["disposition"] = disposition

    fields["incident_id"] = claim_id
    fields["provenance"] = {"evidence_segment_ids": unique_strs([p.segment_id for p in provenance])}

    return Claim(claim_id=claim_id, fields=fields, provenance=provenance)
