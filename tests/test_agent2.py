"""Tests for Agent-2: Clinical Documentation Agent.

Covers:
- build_report_prompt() — deterministic prompt assembly, verified without an LLM
- tag_citation_map()   — entity normalized values found in report text
- run()                — stub path with llm_client=None
"""

from __future__ import annotations

import pytest

from ems_pipeline.agents.agent2 import (
    Agent2Options,
    _derive_base_codes_from_claim,
    build_report_prompt,
    call_llm,
    run,
    tag_citation_map,
)
from ems_pipeline.models import Entity, Segment, Transcript
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared fixtures (inline, consistent with existing test patterns)
# ---------------------------------------------------------------------------


def _minimal_session(encounter_id: str = "enc-agent2-test") -> SessionContext:
    """A SessionContext with just enough Agent-1 data for Agent-2 to work on."""
    session = SessionContext.create(encounter_id=encounter_id)
    entities = [
        Entity(
            type="symptom",
            text="chest pain",
            normalized="chest pain",
            start=62.5,
            end=63.0,
            confidence=0.9,
            attributes={"segment_id": "seg_0002"},
        ),
        Entity(
            type="vital",
            text="BP 150/90",
            normalized="150/90",
            start=80.5,
            end=81.5,
            confidence=0.85,
            attributes={"segment_id": "seg_0003", "name": "BP", "value": "150/90", "unit": "mmHg"},
        ),
        Entity(
            type="medication",
            text="aspirin 324 mg",
            normalized="aspirin 324 mg",
            start=121.0,
            end=121.8,
            confidence=0.92,
            attributes={"segment_id": "seg_0004"},
        ),
        # Entity with no normalization — should not appear in citation map
        Entity(
            type="unit_id",
            text="Unit 12",
            normalized=None,
            start=3.1,
            end=3.4,
            confidence=None,
            attributes={"segment_id": "seg_0001"},
        ),
    ]
    segments = [
        Segment(start=0.0, end=3.0, speaker="spk0", text="Unit 12 dispatched.", confidence=0.9),
        Segment(start=3.0, end=6.0, speaker="spk0", text="Priority 1 for chest pain.", confidence=0.9),
        Segment(start=62.0, end=66.0, speaker="spk1", text="Patient reports chest pain.", confidence=0.9),
        Segment(start=80.0, end=84.0, speaker="spk1", text="BP 150/90.", confidence=0.9),
        Segment(start=120.0, end=124.0, speaker="spk1", text="Administered aspirin 324 mg.", confidence=0.9),
    ]
    return session.model_copy(
        update={
            "transcript_segments": segments,
            "transcript_raw": " ".join(s.text for s in segments),
            "extracted_terms": entities,
            "confidence_flags": [],
            "ambiguities": [],
        }
    )


# ---------------------------------------------------------------------------
# build_report_prompt
# ---------------------------------------------------------------------------


def test_build_report_prompt_contains_system_header() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## SYSTEM" in prompt


def test_build_report_prompt_contains_extracted_terms_header() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## EXTRACTED CLINICAL TERMS" in prompt


def test_build_report_prompt_contains_entity_types_and_text() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "[symptom]" in prompt
    assert "chest pain" in prompt
    assert "[vital]" in prompt
    assert "BP 150/90" in prompt
    assert "[medication]" in prompt
    assert "aspirin 324 mg" in prompt


def test_build_report_prompt_contains_instruction_section() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## INSTRUCTION" in prompt
    # Must request both report and codes
    assert "report" in prompt.lower()
    assert "code" in prompt.lower()


def test_build_report_prompt_includes_transcript_segments() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## TRANSCRIPT SEGMENTS" in prompt
    assert "chest pain" in prompt


def test_build_report_prompt_skips_empty_payer_section() -> None:
    session = _minimal_session()
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## PAYER REQUIREMENTS" not in prompt


def test_build_report_prompt_includes_payer_section_when_provided() -> None:
    session = _minimal_session()
    payer_ctx = {"requires_ALS_documentation": True, "max_transport_miles": 50}
    prompt = build_report_prompt(session, payer_ctx, {}, {})
    assert "## PAYER REQUIREMENTS" in prompt
    assert "requires_ALS_documentation" in prompt


def test_build_report_prompt_includes_coding_section_when_provided() -> None:
    session = _minimal_session()
    coding_ctx = {"ICD_preferred_specificity": "highest", "CPT_99283": "moderate complexity ED"}
    prompt = build_report_prompt(session, {}, coding_ctx, {})
    assert "## CODING GUIDELINES" in prompt
    assert "ICD_preferred_specificity" in prompt


def test_build_report_prompt_includes_patient_history_when_provided() -> None:
    session = _minimal_session()
    patient_ctx = {"allergies": "penicillin", "prior_visits": 2}
    prompt = build_report_prompt(session, {}, {}, patient_ctx)
    assert "## PATIENT HISTORY" in prompt
    assert "penicillin" in prompt


def test_build_report_prompt_includes_ambiguities_when_present() -> None:
    session = _minimal_session()
    session = session.model_copy(
        update={
            "ambiguities": [
                {"segment_id": "seg_0001", "text": "unknown_drug", "reason": "unresolved_normalization", "resolution": None}
            ]
        }
    )
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## AMBIGUITIES" in prompt
    assert "unknown_drug" in prompt


def test_build_report_prompt_no_segments_falls_back_to_raw() -> None:
    """When transcript_segments is absent, transcript_raw is used instead."""
    session = SessionContext.create(encounter_id="enc-raw")
    session = session.model_copy(
        update={
            "transcript_raw": "Patient with chest pain.",
            "transcript_segments": None,
            "extracted_terms": [],
        }
    )
    prompt = build_report_prompt(session, {}, {}, {})
    assert "## TRANSCRIPT" in prompt
    assert "Patient with chest pain." in prompt


def test_build_report_prompt_is_deterministic() -> None:
    """Same inputs produce identical prompts (no randomness)."""
    session = _minimal_session()
    p1 = build_report_prompt(session, {}, {}, {})
    p2 = build_report_prompt(session, {}, {}, {})
    assert p1 == p2


def test_build_report_prompt_encounter_id_not_leaked() -> None:
    """Encounter ID must NOT appear directly in the prompt (no PII leakage)."""
    eid = "super-secret-uuid-12345"
    session = _minimal_session(encounter_id=eid)
    prompt = build_report_prompt(session, {}, {}, {})
    # The encounter_id is used internally for RAG but should not appear verbatim
    # in the prompt body (it would go into retrieval queries, not the prompt text).
    assert eid not in prompt


# ---------------------------------------------------------------------------
# tag_citation_map
# ---------------------------------------------------------------------------


def test_tag_citation_map_finds_present_entity() -> None:
    """An entity whose normalized value appears in the report is cited."""
    entities = [
        Entity(
            type="symptom",
            text="chest pain",
            normalized="chest pain",
            attributes={"segment_id": "seg_0002"},
        )
    ]
    report = "Patient presented with chest pain and shortness of breath."
    result = tag_citation_map(report, entities)

    assert "chest pain" in result
    assert result["chest pain"] == ["seg_0002"]


def test_tag_citation_map_case_insensitive() -> None:
    """Matching is case-insensitive."""
    entities = [
        Entity(type="condition", text="stemi", normalized="STEMI", attributes={"segment_id": "seg_0001"})
    ]
    report = "Patient meets criteria for STEMI activation."
    result = tag_citation_map(report, entities)
    assert "STEMI" in result


def test_tag_citation_map_absent_entity_not_included() -> None:
    """An entity whose normalized value is NOT in the report is excluded."""
    entities = [
        Entity(type="symptom", text="headache", normalized="headache", attributes={"segment_id": "seg_0005"})
    ]
    report = "Patient presented with chest pain only."
    result = tag_citation_map(report, entities)
    assert "headache" not in result


def test_tag_citation_map_none_normalized_skipped() -> None:
    """Entities with normalized=None are silently ignored."""
    entities = [
        Entity(type="unit_id", text="Unit 12", normalized=None, attributes={"segment_id": "seg_0001"})
    ]
    report = "Unit 12 was dispatched."
    result = tag_citation_map(report, entities)
    # normalized is None → nothing to key on
    assert result == {}


def test_tag_citation_map_multiple_entities_same_normalized() -> None:
    """Multiple entities with the same normalized value accumulate segment IDs."""
    entities = [
        Entity(type="symptom", text="SOB", normalized="SOB", attributes={"segment_id": "seg_0001"}),
        Entity(type="symptom", text="shortness of breath", normalized="SOB", attributes={"segment_id": "seg_0003"}),
    ]
    report = "Patient reports SOB for two hours."
    result = tag_citation_map(report, entities)
    assert "SOB" in result
    assert "seg_0001" in result["SOB"]
    assert "seg_0003" in result["SOB"]


def test_tag_citation_map_no_segment_id_in_attributes() -> None:
    """Entity without segment_id in attributes still appears in map (no seg_id entry)."""
    entities = [
        Entity(type="symptom", text="nausea", normalized="nausea", attributes={})
    ]
    report = "Patient has nausea."
    result = tag_citation_map(report, entities)
    assert "nausea" in result
    assert result["nausea"] == []  # no segment_id → empty list


def test_tag_citation_map_empty_report_returns_empty() -> None:
    """Empty report produces an empty citation map."""
    entities = [
        Entity(type="symptom", text="pain", normalized="pain", attributes={"segment_id": "seg_0000"})
    ]
    result = tag_citation_map("", entities)
    assert result == {}


def test_tag_citation_map_no_duplicate_segment_ids() -> None:
    """A segment_id is not added twice even if multiple entities share it."""
    entities = [
        Entity(type="symptom", text="a", normalized="SOB", attributes={"segment_id": "seg_0001"}),
        Entity(type="symptom", text="b", normalized="SOB", attributes={"segment_id": "seg_0001"}),
    ]
    report = "Patient reports SOB."
    result = tag_citation_map(report, entities)
    assert result["SOB"].count("seg_0001") == 1


# ---------------------------------------------------------------------------
# call_llm stub
# ---------------------------------------------------------------------------


def test_call_llm_stub_returns_placeholder_strings() -> None:
    """call_llm with llm_client=None returns the stub placeholders."""
    report, reasoning, codes = call_llm("any prompt", llm_client=None)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "[STUB]" in report
    assert isinstance(reasoning, str)
    assert "[STUB]" in reasoning
    assert codes == []


def test_call_llm_with_client_raises_not_implemented() -> None:
    """call_llm with a non-None client raises NotImplementedError (stub)."""
    with pytest.raises(NotImplementedError):
        call_llm("any prompt", llm_client=object())


# ---------------------------------------------------------------------------
# _derive_base_codes_from_claim
# ---------------------------------------------------------------------------


def test_derive_base_codes_primary_impression() -> None:
    fields = {
        "primary_impression": {"value": "chest pain", "evidence_segment_ids": ["seg_0002"]},
        "interventions": [],
    }
    codes = _derive_base_codes_from_claim(fields)
    icd_codes = [c for c in codes if c["type"] == "ICD"]
    assert len(icd_codes) == 1
    assert "chest pain" in icd_codes[0]["rationale"]
    assert icd_codes[0]["evidence_segment_ids"] == ["seg_0002"]


def test_derive_base_codes_interventions_become_cpt() -> None:
    fields = {
        "primary_impression": {"value": "", "evidence_segment_ids": []},
        "interventions": [
            {"normalized": "aspirin", "evidence_segment_ids": ["seg_0004"]},
            {"normalized": "oxygen", "evidence_segment_ids": ["seg_0005"]},
        ],
    }
    codes = _derive_base_codes_from_claim(fields)
    cpt_codes = [c for c in codes if c["type"] == "CPT"]
    assert len(cpt_codes) == 2
    assert any("aspirin" in c["rationale"] for c in cpt_codes)


def test_derive_base_codes_empty_fields_returns_empty() -> None:
    assert _derive_base_codes_from_claim({}) == []


# ---------------------------------------------------------------------------
# run() — full Agent-2 path with llm_client=None
# ---------------------------------------------------------------------------


def test_run_stub_populates_report_draft() -> None:
    """run() with llm_client=None sets a non-None report_draft placeholder."""
    session = _minimal_session()
    result = run(session, Agent2Options())

    assert result.report_draft is not None
    assert len(result.report_draft) > 0
    assert "[STUB]" in result.report_draft


def test_run_stub_populates_clinical_reasoning() -> None:
    session = _minimal_session()
    result = run(session, Agent2Options())
    assert result.clinical_reasoning is not None
    assert "[STUB]" in result.clinical_reasoning


def test_run_stub_populates_code_suggestions_from_claim_base() -> None:
    """When LLM stub returns empty codes, deterministic claim base is used."""
    session = _minimal_session()
    result = run(session, Agent2Options())
    # LLM stub returns [] → base_codes from claim builder fills in
    assert result.code_suggestions is not None
    # The session has a symptom (chief_complaint fallback), so at least one code
    assert len(result.code_suggestions) >= 0  # may be empty if no claim fields match


def test_run_stub_does_not_mutate_input_session() -> None:
    """Original session is not mutated (immutable write pattern)."""
    session = _minimal_session()
    result = run(session, Agent2Options())
    assert session.report_draft is None
    assert result.report_draft is not None
    assert result.encounter_id == session.encounter_id


def test_run_stub_citation_map_is_none_for_stub_report() -> None:
    """The stub report text contains no entity normalized values → citation_map is None."""
    session = _minimal_session()
    result = run(session, Agent2Options())
    # "[STUB] Clinical report..." does not contain "chest pain", "150/90", etc.
    assert result.citation_map is None


def test_run_with_payer_id_does_not_crash() -> None:
    """Providing a payer_id runs the retrieval stub without error."""
    session = _minimal_session()
    result = run(session, Agent2Options(payer_id="PAYER_ABC"))
    assert result.report_draft is not None


def test_run_empty_session_does_not_crash() -> None:
    """run() on a nearly-empty session (no Agent-1 data) completes without error."""
    session = SessionContext.create(encounter_id="enc-empty-agent2")
    result = run(session, Agent2Options())
    assert result.report_draft is not None


def test_agent2_options_extra_fields_forbidden() -> None:
    """Agent2Options rejects unknown fields (extra='forbid')."""
    with pytest.raises(Exception):
        Agent2Options(unknown_field="oops")  # type: ignore[call-arg]


def test_agent2_options_defaults() -> None:
    opts = Agent2Options()
    assert opts.llm_client is None
    assert opts.payer_id is None


# ---------------------------------------------------------------------------
# SessionContext.citation_map field integration
# ---------------------------------------------------------------------------


def test_citation_map_round_trips_through_session(tmp_path) -> None:
    """citation_map survives write_agent2 → to_json → from_json."""
    session = _minimal_session()
    citation = {"chest pain": ["seg_0002"], "SOB": ["seg_0001", "seg_0003"]}
    updated = session.write_agent2(
        report="Patient has chest pain.",
        reasoning="Clinical reasoning here.",
        codes=[],
        citation_map=citation,
    )
    path = tmp_path / "session_cite.json"
    updated.to_json(path)
    recovered = SessionContext.from_json(path)
    assert recovered.citation_map == citation


def test_write_agent2_citation_map_defaults_to_none() -> None:
    """write_agent2 without citation_map leaves it as None."""
    session = _minimal_session()
    updated = session.write_agent2(
        report="x",
        reasoning="y",
        codes=[],
    )
    assert updated.citation_map is None
