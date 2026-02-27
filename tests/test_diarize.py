from __future__ import annotations

import sys
import types
import warnings
from unittest.mock import MagicMock

import pytest

from ems_pipeline.speech.diarize import (
    DiarizeResult,
    SpeakerTurn,
    diarize,
    validate_speaker_turns,
)
import ems_pipeline.speech.diarize as diarize_mod


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_diarize_fallback_returns_single_turn_with_correct_duration() -> None:
    sr = 16_000
    wave = [0.0] * 40_000  # 2.5 s

    turns = diarize(wave, sr)
    assert len(turns) == 1
    assert turns[0].start == 0.0
    assert turns[0].end == 2.5
    assert turns[0].speaker_id == "spk0"
    assert turns[0].confidence == 1.0


def test_validate_speaker_turns_rejects_overlaps() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="spk0", confidence=0.9),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="spk1", confidence=0.9),
    ]

    with pytest.raises(ValueError, match="overlaps"):
        validate_speaker_turns(turns, duration=3.0, mode="reject")


def test_validate_speaker_turns_normalizes_overlaps() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="spk0", confidence=0.9),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="spk1", confidence=0.9),
    ]

    normalized = validate_speaker_turns(turns, duration=3.0, mode="normalize")
    assert [(t.start, t.end, t.speaker_id) for t in normalized] == [
        (0.0, 2.0, "spk0"),
        (2.0, 3.0, "spk1"),
    ]


# ---------------------------------------------------------------------------
# DiarizeResult backward-compatibility interface
# ---------------------------------------------------------------------------


def test_diarize_returns_diarize_result() -> None:
    """diarize() must return a DiarizeResult, not a bare list."""
    wave = [0.0] * 16_000
    result = diarize(wave, 16_000)
    assert isinstance(result, DiarizeResult)


def test_diarize_result_len_and_indexing() -> None:
    """DiarizeResult supports len() and integer indexing for backward compat."""
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="A"),
        SpeakerTurn(start=1.0, end=2.0, speaker_id="B"),
    ]
    dr = DiarizeResult(turns=turns, backend="test", policy="normalize")

    assert len(dr) == 2
    assert dr[0].speaker_id == "A"
    assert dr[1].speaker_id == "B"


def test_diarize_result_iteration() -> None:
    """DiarizeResult is iterable and yields SpeakerTurn objects."""
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="X")]
    dr = DiarizeResult(turns=turns, backend="b", policy="p")
    collected = list(dr)
    assert collected == turns


def test_diarize_result_turns_property() -> None:
    """DiarizeResult.turns returns the underlying list."""
    turns = [SpeakerTurn(start=0.0, end=1.5, speaker_id="spk0")]
    dr = DiarizeResult(turns=turns, backend="baseline/fallback", policy="reject")
    assert dr.turns is turns


# ---------------------------------------------------------------------------
# Fallback metadata
# ---------------------------------------------------------------------------


def test_diarize_fallback_metadata() -> None:
    """Fallback path must set backend='baseline/fallback' and policy='reject'."""
    wave = [0.0] * 16_000
    result = diarize(wave, 16_000)
    assert result.backend == "baseline/fallback"
    assert result.policy == "reject"


# ---------------------------------------------------------------------------
# Real backend path via monkeypatching _run_optional_backend
# ---------------------------------------------------------------------------


def test_diarize_backend_monkeypatched_multi_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """When _run_optional_backend returns turns, diarize() uses them."""
    fake_turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerTurn(start=1.0, end=2.5, speaker_id="SPEAKER_01"),
    ]

    monkeypatch.setattr(diarize_mod, "_run_optional_backend", lambda w, sr: ("pyannote", fake_turns))
    monkeypatch.setenv("EMS_DIARIZE_OVERLAP_POLICY", "normalize")

    wave = [0.0] * 40_000  # 2.5 s at 16 kHz
    result = diarize(wave, 16_000)

    assert isinstance(result, DiarizeResult)
    assert len(result) == 2
    assert result[0].speaker_id == "SPEAKER_00"
    assert result[1].speaker_id == "SPEAKER_01"
    assert result.backend == "pyannote"
    assert result.policy == "normalize"


def test_diarize_backend_monkeypatched_with_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """normalize policy trims overlapping turns from the backend."""
    overlapping = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="A"),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="B"),  # overlaps by 0.5 s
    ]
    monkeypatch.setattr(
        diarize_mod, "_run_optional_backend", lambda w, sr: ("pyannote", overlapping)
    )
    monkeypatch.setenv("EMS_DIARIZE_OVERLAP_POLICY", "normalize")

    wave = [0.0] * int(3.0 * 16_000)
    result = diarize(wave, 16_000)

    assert [(t.start, t.end) for t in result] == [(0.0, 2.0), (2.0, 3.0)]
    assert result.policy == "normalize"


def test_diarize_backend_unavailable_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """When _run_optional_backend returns None, diarize() falls back to single speaker."""
    monkeypatch.setattr(
        diarize_mod, "_run_optional_backend", lambda w, sr: ("baseline/fallback", None)
    )
    wave = [0.0] * 16_000
    result = diarize(wave, 16_000)

    assert result.backend == "baseline/fallback"
    assert len(result) == 1
    assert result[0].speaker_id == "spk0"


# ---------------------------------------------------------------------------
# _run_optional_backend env-var dispatch
# ---------------------------------------------------------------------------


def test_run_optional_backend_no_env_var_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without EMS_DIARIZE_BACKEND set, _run_optional_backend returns (fallback, None)."""
    monkeypatch.delenv("EMS_DIARIZE_BACKEND", raising=False)
    backend_name, turns = diarize_mod._run_optional_backend([0.0] * 100, 16_000)
    assert backend_name == "baseline/fallback"
    assert turns is None


def test_run_optional_backend_pyannote_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMS_DIARIZE_BACKEND=pyannote dispatches to _run_pyannote_backend."""
    fake_turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="spk0")]
    monkeypatch.setenv("EMS_DIARIZE_BACKEND", "pyannote")
    monkeypatch.setattr(diarize_mod, "_run_pyannote_backend", lambda w, sr: fake_turns)

    backend_name, turns = diarize_mod._run_optional_backend([0.0] * 16_000, 16_000)
    assert backend_name == "pyannote"
    assert turns == fake_turns


def test_run_optional_backend_pyannote_fallback_on_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """If _run_pyannote_backend returns None, falls back to (baseline/fallback, None)."""
    monkeypatch.setenv("EMS_DIARIZE_BACKEND", "pyannote")
    monkeypatch.setattr(diarize_mod, "_run_pyannote_backend", lambda w, sr: None)

    backend_name, turns = diarize_mod._run_optional_backend([0.0] * 16_000, 16_000)
    assert backend_name == "baseline/fallback"
    assert turns is None


# ---------------------------------------------------------------------------
# _run_pyannote_backend: import-error and missing-token paths
# ---------------------------------------------------------------------------


def test_run_pyannote_backend_import_error_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing pyannote.audio emits a RuntimeWarning and returns None."""
    # Mark pyannote.audio as unimportable in this process
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)  # type: ignore[arg-type]
    monkeypatch.setenv("EMS_PYANNOTE_AUTH_TOKEN", "fake_token")

    with pytest.warns(RuntimeWarning, match="not installed"):
        result = diarize_mod._run_pyannote_backend([0.0] * 16_000, 16_000)

    assert result is None


def test_run_pyannote_backend_missing_token_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing auth token emits a RuntimeWarning and returns None."""
    # Pipeline must exist on the fake module so 'from pyannote.audio import Pipeline'
    # succeeds; the token check is what should fire the warning.
    fake_pyannote_pkg = types.ModuleType("pyannote")
    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    monkeypatch.delenv("EMS_PYANNOTE_AUTH_TOKEN", raising=False)

    with pytest.warns(RuntimeWarning, match="EMS_PYANNOTE_AUTH_TOKEN"):
        result = diarize_mod._run_pyannote_backend([0.0] * 16_000, 16_000)

    assert result is None


def test_run_pyannote_backend_runtime_error_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime error from the pipeline emits a RuntimeWarning and returns None."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("model download failed")

    fake_pyannote_pkg = types.ModuleType("pyannote")
    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = mock_pipeline_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    monkeypatch.setenv("EMS_PYANNOTE_AUTH_TOKEN", "tok")

    with pytest.warns(RuntimeWarning, match="diarization failed"):
        result = diarize_mod._run_pyannote_backend([0.0] * 16_000, 16_000)

    assert result is None


def test_run_pyannote_backend_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful pyannote run returns a list of SpeakerTurn objects."""
    # Build mock diarization output
    seg_a = MagicMock()
    seg_a.start = 0.0
    seg_a.end = 1.0
    seg_b = MagicMock()
    seg_b.start = 1.0
    seg_b.end = 2.0

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (seg_a, None, "SPEAKER_00"),
        (seg_b, None, "SPEAKER_01"),
    ]
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_diarization

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    fake_pyannote_pkg = types.ModuleType("pyannote")
    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = mock_pipeline_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    monkeypatch.setenv("EMS_PYANNOTE_AUTH_TOKEN", "tok")

    result = diarize_mod._run_pyannote_backend([0.0] * 32_000, 16_000)

    assert result is not None
    assert len(result) == 2
    assert result[0].speaker_id == "SPEAKER_00"
    assert result[0].start == 0.0
    assert result[0].end == 1.0
    assert result[1].speaker_id == "SPEAKER_01"


# ---------------------------------------------------------------------------
# validate_speaker_turns: "allow" mode
# ---------------------------------------------------------------------------


def test_validate_speaker_turns_allow_passes_overlaps() -> None:
    """allow mode lets overlapping turns through without modification."""
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="spk0", confidence=0.9),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="spk1", confidence=0.9),
    ]
    result = validate_speaker_turns(turns, duration=3.0, mode="allow")

    assert [(t.start, t.end, t.speaker_id) for t in result] == [
        (0.0, 2.0, "spk0"),
        (1.5, 3.0, "spk1"),
    ]


def test_validate_speaker_turns_allow_rejects_negative_start() -> None:
    """allow mode still enforces start >= 0."""
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="spk0")]
    # Manually build an invalid turn bypassing __post_init__ using object.__setattr__
    bad = object.__new__(SpeakerTurn)
    object.__setattr__(bad, "start", -0.5)
    object.__setattr__(bad, "end", 1.0)
    object.__setattr__(bad, "speaker_id", "spk0")
    object.__setattr__(bad, "confidence", 1.0)

    with pytest.raises(ValueError, match="start must be >= 0"):
        validate_speaker_turns([bad], mode="allow")


def test_validate_speaker_turns_allow_rejects_out_of_bounds() -> None:
    """allow mode still enforces end <= duration."""
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker_id="spk0")]
    with pytest.raises(ValueError, match="end must be <= duration"):
        validate_speaker_turns(turns, duration=3.0, mode="allow")


# ---------------------------------------------------------------------------
# validate_speaker_turns: edge cases
# ---------------------------------------------------------------------------


def test_validate_speaker_turns_empty_input() -> None:
    """Empty input returns an empty list regardless of mode."""
    for mode in ("reject", "normalize", "allow"):
        assert validate_speaker_turns([], mode=mode) == []  # type: ignore[arg-type]


def test_validate_speaker_turns_normalize_clamps_to_duration() -> None:
    """normalize mode clamps turns that extend beyond the audio duration."""
    turns = [SpeakerTurn(start=0.0, end=10.0, speaker_id="spk0")]
    result = validate_speaker_turns(turns, duration=5.0, mode="normalize")
    assert len(result) == 1
    assert result[0].end == 5.0


def test_validate_speaker_turns_normalize_drops_zero_duration_after_clamp() -> None:
    """normalize mode drops turns that become empty after clamping."""
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker_id="spk0"),
        SpeakerTurn(start=3.0, end=5.0, speaker_id="spk1"),  # will be clamped to (3, 3)
    ]
    result = validate_speaker_turns(turns, duration=3.0, mode="normalize")
    assert len(result) == 1
    assert result[0].speaker_id == "spk0"


def test_validate_speaker_turns_reject_out_of_bounds() -> None:
    """reject mode raises ValueError when a turn exceeds audio duration."""
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker_id="spk0")]
    with pytest.raises(ValueError, match="end must be <= duration"):
        validate_speaker_turns(turns, duration=3.0, mode="reject")


def test_validate_speaker_turns_sort_by_start() -> None:
    """Turns are sorted by start time regardless of input order."""
    turns = [
        SpeakerTurn(start=2.0, end=3.0, speaker_id="spk1"),
        SpeakerTurn(start=0.0, end=1.0, speaker_id="spk0"),
    ]
    result = validate_speaker_turns(turns, mode="reject")
    assert result[0].start == 0.0
    assert result[1].start == 2.0


# ---------------------------------------------------------------------------
# _get_overlap_policy env-var
# ---------------------------------------------------------------------------


def test_get_overlap_policy_default_is_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMS_DIARIZE_OVERLAP_POLICY", raising=False)
    assert diarize_mod._get_overlap_policy() == "normalize"


def test_get_overlap_policy_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for policy in ("reject", "normalize", "allow"):
        monkeypatch.setenv("EMS_DIARIZE_OVERLAP_POLICY", policy)
        assert diarize_mod._get_overlap_policy() == policy


def test_get_overlap_policy_invalid_falls_back_to_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMS_DIARIZE_OVERLAP_POLICY", "bogus_value")
    assert diarize_mod._get_overlap_policy() == "normalize"
