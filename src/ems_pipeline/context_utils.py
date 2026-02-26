"""Context compression and provenance helpers for stage boundaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from ems_pipeline.models import Entity, ProvenanceLink, Segment

if TYPE_CHECKING:
    from ems_pipeline.session import SessionContext


def _report_excerpt(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def compress_agent1_output(session: SessionContext) -> dict[str, Any]:
    """Build compact Agent-1 boundary context for Agent-2."""
    entity_counts = Counter(ent.type for ent in (session.extracted_terms or []))

    sorted_flags = sorted(
        (session.confidence_flags or []),
        key=lambda flag: float(flag.get("confidence", 1.0)),
    )
    top_flags = sorted_flags[:5]

    speaker_confidences: dict[str, list[float]] = defaultdict(list)
    for segment in session.transcript_segments or []:
        speaker_confidences[segment.speaker].append(segment.confidence)

    transcript_segments_summary = [
        {
            "speaker": speaker,
            "segment_count": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences),
        }
        for speaker, confidences in sorted(speaker_confidences.items())
    ]

    return {
        "encounter_id": session.encounter_id,
        "entity_counts_by_type": dict(sorted(entity_counts.items())),
        "top_confidence_flags": top_flags,
        "ambiguity_count": len(session.ambiguities or []),
        "total_segment_count": len(session.transcript_segments or []),
        "transcript_segments_summary": transcript_segments_summary,
    }


def compress_agent2_output(session: SessionContext) -> dict[str, Any]:
    """Build compact Agent-2 boundary context for Agent-3."""
    reasoning = session.clinical_reasoning
    if reasoning and len(reasoning) > 500:
        reasoning = f"{reasoning[:500]}..."

    return {
        "encounter_id": session.encounter_id,
        "report_draft": session.report_draft,
        "code_suggestions": list(session.code_suggestions or []),
        "clinical_reasoning": reasoning,
        "citation_map_key_count": len(session.citation_map or {}),
    }


def compress_agent3_output(session: SessionContext) -> dict[str, Any]:
    """Build compact Agent-3 boundary context for Agent-4."""
    flags = session.pre_submission_flags or []
    severity_distribution = Counter(
        str(flag.get("severity", "unknown"))
        for flag in flags
        if isinstance(flag, dict)
    )

    compressed: dict[str, Any] = {
        "encounter_id": session.encounter_id,
        "claim_id": session.claim_id,
        "payer_id": session.payer_id,
        "submission_status": session.submission_status,
        "pre_submission_flags_count": len(flags),
        "pre_submission_flags_severity": dict(sorted(severity_distribution.items())),
        "report_excerpt": (session.report_draft or "")[:300],
    }
    if session.denial_reason:
        compressed["denial_reason"] = session.denial_reason
    return compressed


def tag_citations(
    text: str,
    entities: list[Entity],
    segments: list[Segment],
) -> dict[str, list[str]]:
    """Tag normalized entities in text with supporting segment IDs."""
    _ = segments
    if not text:
        return {}

    text_lower = text.lower()
    citation_map: dict[str, list[str]] = {}

    for entity in entities:
        normalized = entity.normalized
        if normalized is None:
            continue
        if normalized.lower() not in text_lower:
            continue

        segment_id = (
            entity.attributes.get("segment_id", "unknown")
            if isinstance(entity.attributes, dict)
            else "unknown"
        )
        if not isinstance(segment_id, str) or not segment_id:
            segment_id = "unknown"

        if normalized not in citation_map:
            citation_map[normalized] = []
        if segment_id not in citation_map[normalized]:
            citation_map[normalized].append(segment_id)

    return citation_map


def _code_report_excerpt(report_draft: str | None, code_entry: dict[str, Any]) -> str:
    if not report_draft:
        return ""

    report_lower = report_draft.lower()
    candidates = [
        str(code_entry.get("code") or "").strip(),
        str(code_entry.get("rationale") or "").strip(),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        idx = report_lower.find(candidate.lower())
        if idx < 0:
            continue
        start = max(0, idx - 80)
        end = min(len(report_draft), idx + max(len(candidate), 120))
        excerpt = report_draft[start:end]
        if start > 0:
            excerpt = f"...{excerpt}"
        if end < len(report_draft):
            excerpt = f"{excerpt}..."
        return excerpt

    return _report_excerpt(report_draft, 220)


def build_provenance_chain(session: SessionContext) -> list[dict[str, Any]]:
    """Build claim-code → report → segment → entity provenance entries."""
    chain: list[dict[str, Any]] = []
    extracted_terms = session.extracted_terms or []
    citation_map = session.citation_map or {}

    for code_entry in session.code_suggestions or []:
        if not isinstance(code_entry, dict):
            continue

        evidence_ids = code_entry.get("evidence_segment_ids")
        segment_ids: list[str] = []
        if isinstance(evidence_ids, list):
            segment_ids = [
                seg_id for seg_id in evidence_ids if isinstance(seg_id, str) and seg_id
            ]

        if not segment_ids and citation_map:
            rationale = str(code_entry.get("rationale") or "").lower()
            for normalized, normalized_segment_ids in citation_map.items():
                if normalized.lower() not in rationale:
                    continue
                for seg_id in normalized_segment_ids:
                    if seg_id not in segment_ids:
                        segment_ids.append(seg_id)

        links = [ProvenanceLink(segment_id=seg_id) for seg_id in segment_ids]
        linked_segment_ids = [link.segment_id for link in links]

        entities: list[dict[str, Any]] = []
        seen_entities: set[tuple[str, str, str | None]] = set()
        for entity in extracted_terms:
            seg_id = (
                entity.attributes.get("segment_id")
                if isinstance(entity.attributes, dict)
                else None
            )
            if seg_id not in linked_segment_ids:
                continue
            signature = (entity.type, entity.text, seg_id)
            if signature in seen_entities:
                continue
            seen_entities.add(signature)
            entities.append(
                {
                    "type": entity.type,
                    "text": entity.text,
                    "normalized": entity.normalized,
                    "segment_id": seg_id,
                }
            )

        chain.append(
            {
                "code": code_entry.get("code"),
                "code_type": code_entry.get("type"),
                "report_excerpt": _code_report_excerpt(
                    session.report_draft,
                    code_entry,
                ),
                "segment_ids": linked_segment_ids,
                "entities": entities,
            }
        )

    return chain
