from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ems_pipeline.audio.preprocess import TARGET_SAMPLE_RATE, chunk_audio, load_audio


def _write_wav_int16(path: Path, data: np.ndarray, sr: int) -> None:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        frames = data[:, None]
    elif data.ndim == 2:
        frames = data
    else:
        raise ValueError("data must be 1D or 2D.")

    frames = np.clip(frames, -1.0, 1.0)
    pcm = (frames * 32767.0).round().astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(frames.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes(order="C"))


def test_load_audio_mixdown_stereo(tmp_path: Path) -> None:
    sr = TARGET_SAMPLE_RATE
    t = np.arange(sr, dtype=np.float32) / sr
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    stereo = np.stack([left, right], axis=1)

    wav_path = tmp_path / "stereo.wav"
    _write_wav_int16(wav_path, stereo, sr=sr)

    y = load_audio(wav_path)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.shape[0] == sr

    expected = (left + right) / 2.0
    np.testing.assert_allclose(y, expected, atol=2e-4, rtol=0)


def test_load_audio_resamples_to_16khz(tmp_path: Path) -> None:
    sr_in = 48_000
    duration_s = 1.0
    n = int(sr_in * duration_s)
    t = np.arange(n, dtype=np.float32) / sr_in
    mono = 0.5 * np.sin(2 * np.pi * 1000.0 * t)

    wav_path = tmp_path / "mono_48k.wav"
    _write_wav_int16(wav_path, mono, sr=sr_in)

    y = load_audio(wav_path)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.shape[0] == TARGET_SAMPLE_RATE


def test_chunk_audio_count_and_offsets() -> None:
    sr = TARGET_SAMPLE_RATE
    duration_s = 65
    wave_ = np.zeros((duration_s * sr,), dtype=np.float32)

    chunks = chunk_audio(wave_, sr=sr, chunk_seconds=30, overlap_seconds=2)
    assert len(chunks) == 3

    t0s = [t0 for _, t0 in chunks]
    assert t0s == [0.0, 28.0, 56.0]

    chunk_lens = [c.shape[0] for c, _ in chunks]
    assert chunk_lens[0] == 30 * sr
    assert chunk_lens[1] == 30 * sr
    assert chunk_lens[2] == (duration_s - 56) * sr

