from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TARGET_SAMPLE_RATE = 16_000
SAFE_PEAK = 0.99


@dataclass(frozen=True)
class AudioData:
    wave: np.ndarray  # float32, shape (n,) mono
    sample_rate: int


def _read_wav_pcm(path: Path) -> AudioData:
    import wave

    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        # 8-bit PCM is unsigned.
        data_u8 = np.frombuffer(raw, dtype=np.uint8)
        data_i16 = data_u8.astype(np.int16) - 128
        data = (data_i16.astype(np.float32)) / 128.0
    elif sample_width == 2:
        data_i16 = np.frombuffer(raw, dtype="<i2")
        data = data_i16.astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit little-endian signed PCM: unpack into int32 with sign-extension.
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size % 3 != 0:
            raise ValueError("Invalid 24-bit PCM WAV (byte count not divisible by 3).")
        b = b.reshape(-1, 3)
        x = (b[:, 0].astype(np.int32)) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        x = (x << 8) >> 8
        data = x.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        # Commonly 32-bit signed PCM. (32-bit float WAV is not supported via stdlib wave.)
        data_i32 = np.frombuffer(raw, dtype="<i4")
        data = data_i32.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if num_channels <= 0:
        raise ValueError("Invalid WAV: non-positive channel count.")

    if data.size % num_channels != 0:
        raise ValueError("Invalid WAV: sample count not divisible by channel count.")

    frames = data.reshape(-1, num_channels)
    if num_channels == 1:
        mono = frames[:, 0]
    else:
        mono = frames.mean(axis=1)

    return AudioData(wave=np.asarray(mono, dtype=np.float32), sample_rate=int(sample_rate))


def _resample_linear(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wave
    if wave.size == 0:
        return wave.astype(np.float32, copy=False)
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("Sample rates must be positive.")

    n_in = int(wave.shape[0])
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_out <= 0:
        return np.zeros((0,), dtype=np.float32)

    x_old = np.arange(n_in, dtype=np.float64) / float(sr_in)
    x_new = np.arange(n_out, dtype=np.float64) / float(sr_out)
    y_new = np.interp(x_new, x_old, wave.astype(np.float64))
    return y_new.astype(np.float32)


def _safe_peak_normalize(wave: np.ndarray, safe_peak: float = SAFE_PEAK) -> np.ndarray:
    if wave.size == 0:
        return wave.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(wave)))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if peak <= safe_peak:
        return wave.astype(np.float32, copy=False)
    scale = safe_peak / peak
    return (wave * scale).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """
    Load an audio file and return a mono float32 waveform at 16 kHz.

    - If input is stereo/multi-channel, it is mixed down by averaging channels.
    - If input sample rate != 16 kHz, it is resampled using linear interpolation.
    - If the waveform peak exceeds `SAFE_PEAK`, it is scaled down to avoid clipping.

    Notes:
    - This function supports PCM WAV via the Python standard library.
    - If `soundfile` is installed, it will be used to support additional formats.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    audio: AudioData | None = None

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        # data: (frames, channels)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        audio = AudioData(wave=np.asarray(mono, dtype=np.float32), sample_rate=int(sr))
    except ImportError:
        audio = None
    except Exception:
        # Fall back to WAV PCM reader when soundfile can't decode the file.
        audio = None

    if audio is None:
        if p.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(
                f"Unsupported audio format {p.suffix!r} without optional dependency 'soundfile'."
            )
        audio = _read_wav_pcm(p)

    wave = audio.wave
    sr_in = int(audio.sample_rate)

    wave_16k = _resample_linear(wave, sr_in=sr_in, sr_out=TARGET_SAMPLE_RATE)
    wave_16k = _safe_peak_normalize(wave_16k, safe_peak=SAFE_PEAK)
    return np.ascontiguousarray(wave_16k, dtype=np.float32)


def chunk_audio(
    wave: np.ndarray,
    sr: int,
    chunk_seconds: float = 30,
    overlap_seconds: float = 2,
) -> list[tuple[np.ndarray, float]]:
    """
    Split a mono waveform into overlapping chunks.

    Returns a list of `(chunk_wave, t0_seconds)` where `t0_seconds` is the start time of the chunk.

    The final chunk may be shorter than `chunk_seconds`.
    """

    if sr <= 0:
        raise ValueError("sr must be positive.")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative.")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds.")

    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1:
        raise ValueError("wave must be a 1D mono array.")
    if wave.size == 0:
        return []

    chunk_len = int(round(chunk_seconds * sr))
    overlap_len = int(round(overlap_seconds * sr))
    if chunk_len <= 0:
        raise ValueError("chunk_seconds too small for given sr.")
    step = chunk_len - overlap_len
    if step <= 0:
        raise ValueError("overlap_seconds too large for given chunk_seconds and sr.")

    chunks: list[tuple[np.ndarray, float]] = []
    n = int(wave.shape[0])
    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        chunk = wave[start:end]
        chunks.append((np.ascontiguousarray(chunk, dtype=np.float32), start / float(sr)))
        if end >= n:
            break
        start += step
    return chunks


def _iter_chunks(
    wave: np.ndarray, sr: int, chunk_seconds: float, overlap_seconds: float
) -> Iterable[tuple[np.ndarray, float]]:
    # Internal generator for streaming use-cases.
    for chunk, t0 in chunk_audio(wave, sr, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds):
        yield chunk, t0

