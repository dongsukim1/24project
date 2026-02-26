"""Build a structured proto-claim from transcript + extracted artifacts.

This module is intentionally rules-based and deterministic (MVP). It stitches
together:
- `Transcript` segments (for stable evidence segment IDs)
- extracted `Entity` mentions
- derived timeline `Event`s (operational context)
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypedDict

from ems_pipeline.models import Claim, Entity, ProvenanceLink, Transcript


class _Event(TypedDict):
    type: str
    start: float
    end: float
    speaker: str | None
    evidence_segment_ids: list[str]
    payload: dict[str, Any]


DispositionStatus = Literal["transported", "refused", "canceled", "unknown"]


def _segment_ids(transcript: Transcript) -> list[str]:
    seg_id_map = transcript.metadata.get("segment_id_map")
    if isinstance(seg_id_map, dict) and seg_id_map:
        inv: dict[int, str] = {}
        for seg_id, idx in seg_id_map.items():
            if isinstance(seg_id, str) and isinstance(idx, int):
                inv[idx] = seg_id
        return [inv.get(i, f"seg_{i:04d}") for i in range(len(transcript.segments))]
    return [f"seg_{i:04d}" for i in range(len(transcript.segments))]


def _evidence_segment_ids_for_span(transcript: Transcript, start: float | None, end: float | None) -> list[str]:
    if start is None and end is None:
        return []
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return []
    if end < start:
        start, end = end, start

    segment_ids = _segment_ids(transcript)
    overlaps: list[str] = []
    for i, seg in enumerate(transcript.segments):
        if seg.end >= start and seg.start <= end:
            overlaps.append(segment_ids[i])
    return overlaps


def _evidence_segment_ids_for_text(transcript: Transcript, text: str) -> list[str]:
    needle = (text or "").strip().lower()
    if not needle:
        return []
    segment_ids = _segment_ids(transcript)
    hits: list[str] = []
    for i, seg in enumerate(transcript.segments):
        if needle in seg.text.lower():
            hits.append(segment_ids[i])
    return hits


def _evidence_for_entity(transcript: Transcript, ent: Entity) -> list[str]:
    by_span = _evidence_segment_ids_for_span(transcript, ent.start, ent.end)
    if by_span:
        return by_span
    return _evidence_segment_ids_for_text(transcript, ent.text)


def _unique_strs(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _audio_filename_from_transcript(transcript: Transcript) -> str:
    for key in ("audio_filename", "audio_file", "audio_path", "source"):
        raw = transcript.metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return os.path.basename(raw.strip())
    return "unknown_audio"


def _incident_id(audio_filename: str, start_seconds: float) -> str:
    key = f"{audio_filename}:{start_seconds:.3f}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _value_with_evidence(
    value: Any,
    *,
    evidence_segment_ids: Iterable[str],
    evidence_entity_indexes: Iterable[int] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "value": value,
        "evidence_segment_ids": _unique_strs(evidence_segment_ids),
    }
    entity_indexes = sorted({int(i) for i in evidence_entity_indexes})
    if entity_indexes:
        payload["evidence_entity_indexes"] = entity_indexes
    return payload


def _first_entity(
    entities: Sequence[Entity],
    types: set[str],
) -> tuple[int, Entity] | None:
    for i, ent in enumerate(entities):
        if (ent.type or "").strip().lower() in types:
            return i, ent
    return None


def _all_entities(
    entities: Sequence[Entity],
    types: set[str],
) -> list[tuple[int, Entity]]:
    out: list[tuple[int, Entity]] = []
    for i, ent in enumerate(entities):
        if (ent.type or "").strip().lower() in types:
            out.append((i, ent))
    return out


def _events_by_type(events: Sequence[_Event], event_type: str) -> list[_Event]:
    return [e for e in events if (e.get("type") or "") == event_type]


def build_claim(
    transcript: Transcript,
    entities: Sequence[Entity],
    events: Sequence[_Event],
) -> tuple[Claim, dict[str, Any]]:
    """Build a proto-claim with evidence segment links.

    Returns:
        (Claim, claim_json_dict)
    """

    audio_filename = _audio_filename_from_transcript(transcript)
    start_seconds = float(transcript.segments[0].start) if transcript.segments else 0.0
    incident_id = _incident_id(audio_filename, start_seconds)

    fields: dict[str, Any] = {}
    provenance_links: list[ProvenanceLink] = []

    def add_provenance(segment_ids: Iterable[str], note: str) -> None:
        for seg_id in _unique_strs(segment_ids):
            provenance_links.append(ProvenanceLink(segment_id=seg_id, note=note))

    # location_hint
    loc = _first_entity(entities, {"location", "address", "cross_street"})
    if loc is not None:
        i, ent = loc
        seg_ids = _evidence_for_entity(transcript, ent)
        fields["location_hint"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "location_hint")
    else:
        fields["location_hint"] = _value_with_evidence("", evidence_segment_ids=[])

    # dispatch
    dispatch: dict[str, Any] = {}
    dispatch_evidence: list[str] = []

    call_events = _events_by_type(events, "CALL_RECEIVED")
    dispatched_events = _events_by_type(events, "DISPATCHED")
    for ev in [*call_events, *dispatched_events]:
        dispatch_evidence.extend(ev.get("evidence_segment_ids") or [])

    inc = _first_entity(entities, {"incident_type", "call_type", "dispatch_reason"})
    if inc is not None:
        i, ent = inc
        seg_ids = _evidence_for_entity(transcript, ent)
        dispatch_evidence.extend(seg_ids)
        dispatch["incident_type"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "dispatch.incident_type")
    else:
        dispatch["incident_type"] = _value_with_evidence("", evidence_segment_ids=dispatch_evidence)

    pr = _first_entity(entities, {"priority", "acuity"})
    if pr is not None:
        i, ent = pr
        seg_ids = _evidence_for_entity(transcript, ent)
        dispatch_evidence.extend(seg_ids)
        dispatch["priority"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "dispatch.priority")
    else:
        dispatch["priority"] = _value_with_evidence("", evidence_segment_ids=dispatch_evidence)

    unit_entities = _all_entities(entities, {"unit", "unit_id"})
    units: list[dict[str, Any]] = []
    for i, ent in unit_entities:
        seg_ids = _evidence_for_entity(transcript, ent)
        dispatch_evidence.extend(seg_ids)
        units.append(
            _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        )
        add_provenance(seg_ids, "dispatch.units")
    dispatch["units"] = units

    dispatch["evidence_segment_ids"] = _unique_strs(dispatch_evidence)
    fields["dispatch"] = dispatch

    # patient (optional)
    patient: dict[str, Any] = {}
    age = _first_entity(entities, {"age", "patient_age"})
    sex = _first_entity(entities, {"sex", "gender", "patient_sex"})
    if age is not None:
        i, ent = age
        seg_ids = _evidence_for_entity(transcript, ent)
        patient["age_hint"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "patient.age_hint")
    if sex is not None:
        i, ent = sex
        seg_ids = _evidence_for_entity(transcript, ent)
        patient["sex_hint"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "patient.sex_hint")
    if patient:
        fields["patient"] = patient

    # primary_impression / chief_complaint
    cc = _first_entity(entities, {"chief_complaint", "complaint", "primary_impression"})
    if cc is None:
        cc = _first_entity(entities, {"symptom"})
    if cc is not None:
        i, ent = cc
        seg_ids = _evidence_for_entity(transcript, ent)
        fields["primary_impression"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "primary_impression")
    else:
        fields["primary_impression"] = _value_with_evidence("", evidence_segment_ids=[])

    # findings: symptoms + vitals
    findings: dict[str, Any] = {}

    symptoms: list[dict[str, Any]] = []
    for i, ent in _all_entities(entities, {"symptom"}):
        seg_ids = _evidence_for_entity(transcript, ent)
        symptoms.append(_value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i]))
        add_provenance(seg_ids, "findings.symptoms")
    findings["symptoms"] = symptoms

    vitals: list[dict[str, Any]] = []
    for i, ent in _all_entities(entities, {"vital", "vitals"}):
        seg_ids = _evidence_for_entity(transcript, ent)
        name = ent.attributes.get("name") if isinstance(ent.attributes, dict) else None
        vital_item: dict[str, Any] = {
            "name": name or (ent.normalized or ent.text),
            "value": ent.attributes.get("value") if isinstance(ent.attributes, dict) else None,
            "unit": ent.attributes.get("unit") if isinstance(ent.attributes, dict) else None,
            "evidence_segment_ids": _unique_strs(seg_ids),
            "evidence_entity_indexes": [i],
        }
        # Drop nulls for a cleaner JSON surface.
        vital_item = {k: v for k, v in vital_item.items() if v not in (None, "", [])}
        vitals.append(vital_item)
        add_provenance(seg_ids, "findings.vitals")
    findings["vitals"] = vitals

    fields["findings"] = findings

    # interventions: meds/procedures
    interventions: list[dict[str, Any]] = []
    for i, ent in _all_entities(entities, {"medication", "procedure", "intervention", "treatment"}):
        seg_ids = _evidence_for_entity(transcript, ent)
        kind = (ent.type or "").strip().lower()
        interventions.append(
            {
                "kind": kind,
                "text": ent.text,
                "normalized": ent.normalized,
                "evidence_segment_ids": _unique_strs(seg_ids),
                "evidence_entity_indexes": [i],
            }
        )
        add_provenance(seg_ids, "interventions")
    fields["interventions"] = interventions

    # disposition
    disposition: dict[str, Any] = {}
    disp_evidence: list[str] = []

    # Status inference from events + entities.
    status: DispositionStatus = "unknown"
    if _events_by_type(events, "TRANSPORT_BEGIN") or _events_by_type(events, "ARRIVE_ED"):
        status = "transported"
        for ev in [*_events_by_type(events, "TRANSPORT_BEGIN"), *_events_by_type(events, "ARRIVE_ED")]:
            disp_evidence.extend(ev.get("evidence_segment_ids") or [])

    disp_ent = _first_entity(entities, {"disposition"})
    if disp_ent is not None:
        i, ent = disp_ent
        seg_ids = _evidence_for_entity(transcript, ent)
        disp_evidence.extend(seg_ids)
        norm = (ent.normalized or ent.text or "").strip().lower()
        if any(k in norm for k in ("refus", "ama")):
            status = "refused"
        if any(k in norm for k in ("cancel", "canceled", "cancelled")):
            status = "canceled"
        disposition["status"] = _value_with_evidence(status, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "disposition.status")
    else:
        disposition["status"] = _value_with_evidence(status, evidence_segment_ids=disp_evidence)

    dest_ent = _first_entity(entities, {"destination", "destination_hint", "hospital"})
    if dest_ent is not None:
        i, ent = dest_ent
        seg_ids = _evidence_for_entity(transcript, ent)
        disp_evidence.extend(seg_ids)
        disposition["destination_hint"] = _value_with_evidence(ent.normalized or ent.text, evidence_segment_ids=seg_ids, evidence_entity_indexes=[i])
        add_provenance(seg_ids, "disposition.destination_hint")
    else:
        disposition["destination_hint"] = _value_with_evidence("", evidence_segment_ids=disp_evidence)

    disposition["evidence_segment_ids"] = _unique_strs(disp_evidence)
    fields["disposition"] = disposition

    # incident_id always present
    fields["incident_id"] = incident_id

    # top-level provenance summary (MVP)
    used_segments = _unique_strs([p.segment_id for p in provenance_links])
    fields["provenance"] = {"evidence_segment_ids": used_segments}

    claim = Claim(claim_id=incident_id, fields=fields, provenance=provenance_links)
    claim_json = claim.model_dump(mode="json")
    return claim, claim_json

