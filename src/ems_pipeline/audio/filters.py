"""Lightweight audio filters.

Design goals:
- No third-party dependencies (pure Python).
- Good-enough signal conditioning for speech-centric use cases.

All filters operate on mono float samples (typically in [-1.0, 1.0]).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Final


def _as_floats(wave: Sequence[float] | Iterable[float]) -> list[float]:
    if isinstance(wave, list):
        return wave
    return [float(x) for x in wave]


@dataclass(frozen=True)
class _Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def _biquad_lowpass(sr: int, fc: float, q: float = 1.0 / math.sqrt(2.0)) -> _Biquad:
    omega = 2.0 * math.pi * (fc / sr)
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2.0 * q)

    b0 = (1.0 - cos_omega) / 2.0
    b1 = 1.0 - cos_omega
    b2 = (1.0 - cos_omega) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha

    inv_a0 = 1.0 / a0
    return _Biquad(b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0)


def _biquad_highpass(sr: int, fc: float, q: float = 1.0 / math.sqrt(2.0)) -> _Biquad:
    omega = 2.0 * math.pi * (fc / sr)
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2.0 * q)

    b0 = (1.0 + cos_omega) / 2.0
    b1 = -(1.0 + cos_omega)
    b2 = (1.0 + cos_omega) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha

    inv_a0 = 1.0 / a0
    return _Biquad(b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0)


def _apply_biquad(wave: Sequence[float], biquad: _Biquad) -> list[float]:
    y: list[float] = [0.0] * len(wave)
    x1 = x2 = 0.0
    y1 = y2 = 0.0

    b0, b1, b2, a1, a2 = biquad.b0, biquad.b1, biquad.b2, biquad.a1, biquad.a2
    for i, x0 in enumerate(wave):
        y0 = (b0 * x0) + (b1 * x1) + (b2 * x2) - (a1 * y1) - (a2 * y2)
        y[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return y


def bandpass_filter(wave: Sequence[float] | Iterable[float], sr: int, low: float = 200, high: float = 3400) -> list[float]:
    """Telephone-like bandpass filter (defaults ~200–3400 Hz).

    Implementation: cascaded 4th-order highpass + 4th-order lowpass using RBJ biquads.
    """

    if sr <= 0:
        raise ValueError("sr must be > 0")
    if not (0 < low < high < (sr / 2)):
        raise ValueError("Expected 0 < low < high < sr/2")

    samples = _as_floats(wave)

    hp = _biquad_highpass(sr, low)
    lp = _biquad_lowpass(sr, high)

    # Apply twice each for a steeper (speech-like) rolloff.
    out = _apply_biquad(samples, hp)
    out = _apply_biquad(out, hp)
    out = _apply_biquad(out, lp)
    out = _apply_biquad(out, lp)
    return out


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fft_inplace(a: list[complex], invert: bool) -> None:
    """Iterative radix-2 FFT in-place."""

    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("FFT length must be a power of two")

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    sign = -1.0 if not invert else 1.0
    while length <= n:
        angle = sign * (2.0 * math.pi / length)
        wlen = complex(math.cos(angle), math.sin(angle))
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            half = length // 2
            for j in range(i, i + half):
                u = a[j]
                v = a[j + half] * w
                a[j] = u + v
                a[j + half] = u - v
                w *= wlen
        length <<= 1

    if invert:
        inv_n = 1.0 / n
        for i in range(n):
            a[i] *= inv_n


def _hann(n: int) -> list[float]:
    if n <= 1:
        return [1.0] * max(n, 0)
    return [0.5 - 0.5 * math.cos((2.0 * math.pi * i) / (n - 1)) for i in range(n)]


def denoise(
    wave: Sequence[float] | Iterable[float],
    sr: int,
    *,
    frame_ms: float = 32.0,
    hop_ms: float = 8.0,
    noise_estimate_ms: float = 250.0,
    threshold_mul: float = 1.5,
    floor_gain: float = 0.1,
) -> list[float]:
    """Simple spectral-gate noise reduction (pure Python).

    Notes:
    - Estimates noise from the first `noise_estimate_ms` of audio.
    - Applies a soft gate by scaling quiet bins to `floor_gain`.
    """

    if sr <= 0:
        raise ValueError("sr must be > 0")
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("frame_ms and hop_ms must be > 0")
    if not (0.0 < floor_gain <= 1.0):
        raise ValueError("floor_gain must be in (0, 1]")
    if threshold_mul <= 0:
        raise ValueError("threshold_mul must be > 0")

    samples = _as_floats(wave)
    n = len(samples)
    if n == 0:
        return []

    frame_len = max(64, int(sr * frame_ms / 1000.0))
    # Next power of two for FFT efficiency + our radix-2 implementation.
    nfft = 1
    while nfft < frame_len:
        nfft <<= 1
    hop = max(16, int(sr * hop_ms / 1000.0))

    window = _hann(frame_len)
    window_sq = [w * w for w in window]

    # Output buffer + normalization (for overlap-add with window).
    out: list[float] = [0.0] * (n + nfft)
    norm: list[float] = [0.0] * (n + nfft)

    # Determine how many frames to use for noise estimation.
    noise_samples = min(n, int(sr * noise_estimate_ms / 1000.0))
    noise_frames = max(1, (max(0, noise_samples - frame_len) // hop) + 1)

    half = nfft // 2
    noise_mag: list[float] = [0.0] * (half + 1)
    noise_count = 0

    def frame_at(start: int) -> list[complex]:
        buf = [0.0j] * nfft
        end = min(n, start + frame_len)
        for i in range(start, end):
            buf[i - start] = complex(samples[i] * window[i - start], 0.0)
        return buf

    # 1) Noise profile (average magnitude spectrum over initial frames).
    pos = 0
    for _ in range(noise_frames):
        buf = frame_at(pos)
        _fft_inplace(buf, invert=False)
        for k in range(half + 1):
            noise_mag[k] += abs(buf[k])
        noise_count += 1
        pos += hop
        if pos + frame_len > n:
            break

    if noise_count:
        inv = 1.0 / noise_count
        for k in range(half + 1):
            noise_mag[k] *= inv

    # 2) Gate frames.
    pos = 0
    while pos < n:
        buf = frame_at(pos)
        _fft_inplace(buf, invert=False)

        # Apply symmetric gains to preserve real-valued output.
        for k in range(half + 1):
            mag = abs(buf[k])
            thresh = noise_mag[k] * threshold_mul
            gain = 1.0 if mag >= thresh else floor_gain
            buf[k] *= gain
            if 0 < k < half:
                buf[nfft - k] *= gain

        _fft_inplace(buf, invert=True)

        # Overlap-add + normalization.
        end = min(n, pos + frame_len)
        for i in range(pos, end):
            wi = i - pos
            out[i] += float(buf[wi].real) * window[wi]
            norm[i] += window_sq[wi]

        pos += hop

    # Normalize to account for window overlap.
    # Avoid amplifying leading/trailing samples where the window energy is tiny.
    min_denom: Final[float] = 1e-3
    result: list[float] = [0.0] * n
    for i in range(n):
        denom = norm[i]
        result[i] = out[i] / denom if denom >= min_denom else out[i]

    return result
