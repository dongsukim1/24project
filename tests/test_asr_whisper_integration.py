from __future__ import annotations

import os

import pytest

from ems_pipeline.speech.asr_whisper import asr_items_to_segments, transcribe_chunks


@pytest.mark.integration
def test_smoke_transcribe_chunks_runs_offline_local():
    # This is intentionally a very small test and can be skipped in CI.
    #
    # It only checks that the code path works end-to-end *if* optional dependencies and a local
    # model cache are present. It does not assert on transcript contents.
    try:
        import numpy as np  # type: ignore
    except Exception:
        pytest.skip("numpy is required for the integration smoke test")

    # If the user hasn't installed a backend, skip.
    try:
        import faster_whisper  # type: ignore  # noqa: F401
    except Exception:
        try:
            import whisper  # type: ignore  # noqa: F401
        except Exception:
            pytest.skip("No whisper backend installed")

    if os.getenv("CI") or os.getenv("EMS_SKIP_INTEGRATION_TESTS"):
        pytest.skip("Skipping integration tests in CI.")

    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    items = transcribe_chunks([silence], sr=sr, language="en")
    segs = asr_items_to_segments(items, speaker="spk0")
    assert isinstance(segs, list)

