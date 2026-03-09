from __future__ import annotations

from unittest.mock import MagicMock

import ems_pipeline.extract as extract_module
from ems_pipeline.extract import extract_entities
from ems_pipeline.llm.extractor import LlmExtractor
from ems_pipeline.models import EntitiesDocument, Segment, Transcript


def _make_mock_response(entities_payload: list[dict]) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "record_entities"
    tool_block.input = {"entities": entities_payload}
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _keys(entities_doc: EntitiesDocument) -> set[tuple[str, str]]:
    return {(e.type, (e.normalized or e.text)) for e in entities_doc.entities}


def _setup_mock_extractor(mock_client: MagicMock) -> None:
    """Inject a mock-backed LlmExtractor into the extract module."""
    extractor = LlmExtractor.__new__(LlmExtractor)
    extractor._client = mock_client
    extractor._model = "claude-sonnet-4-20250514"
    extract_module._llm_extractor = extractor


def test_extract_entities_basic() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        {"type": "VITAL_SPO2", "text": "SpO2 88%", "normalized": "SpO2 88%",
         "segment_id": "seg_0000", "confidence": 0.95},
        {"type": "SYMPTOM", "text": "SOB", "normalized": "SOB",
         "segment_id": "seg_0000", "confidence": 0.9},
        {"type": "CONDITION", "text": "STEMI", "normalized": "STEMI",
         "segment_id": "seg_0000", "confidence": 0.92},
    ])

    _setup_mock_extractor(mock_client)

    transcript = Transcript(segments=[
        Segment(start=0.0, end=10.0, speaker="spk0",
                text="SpO2 88% patient has SOB STEMI alert", confidence=0.92),
    ])

    result = extract_entities(transcript)

    assert isinstance(result, EntitiesDocument)
    keys = _keys(result)
    assert ("VITAL_SPO2", "SpO2 88%") in keys
    assert ("SYMPTOM", "SOB") in keys
    assert ("CONDITION", "STEMI") in keys


def test_extract_entities_populates_segment_metadata() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([
        {"type": "SYMPTOM", "text": "chest pain", "segment_id": "seg_0000",
         "confidence": 0.9},
    ])

    _setup_mock_extractor(mock_client)

    transcript = Transcript(segments=[
        Segment(start=5.0, end=15.0, speaker="medic1",
                text="patient reports chest pain", confidence=0.85),
    ])

    result = extract_entities(transcript)

    entity = result.entities[0]
    assert entity.start == 5.0
    assert entity.end == 15.0
    assert entity.speaker == "medic1"
    assert entity.attributes["source"] == "llm"


def test_extract_entities_debug_mode(capsys) -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response([])

    _setup_mock_extractor(mock_client)

    transcript = Transcript(segments=[
        Segment(start=0.0, end=1.0, speaker="spk0", text="hello", confidence=0.9),
    ])

    extract_entities(transcript, debug=True)

    captured = capsys.readouterr()
    assert "[extract.debug]" in captured.err
