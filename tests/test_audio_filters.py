from __future__ import annotations

import math
import random

from ems_pipeline.audio.filters import bandpass_filter, denoise


def _rms(x: list[float]) -> float:
    if not x:
        return 0.0
    return math.sqrt(sum(v * v for v in x) / len(x))


def _sine(sr: int, freq: float, seconds: float, amp: float = 0.5) -> list[float]:
    n = int(sr * seconds)
    two_pi_f = 2.0 * math.pi * freq
    return [amp * math.sin(two_pi_f * (i / sr)) for i in range(n)]


def test_bandpass_preserves_1khz_attenuates_low_and_high() -> None:
    sr = 44100
    seconds = 1.0

    x_1k = _sine(sr, 1000.0, seconds)
    x_50 = _sine(sr, 50.0, seconds)
    x_8k = _sine(sr, 8000.0, seconds)

    y_1k = bandpass_filter(x_1k, sr, low=200, high=3400)
    y_50 = bandpass_filter(x_50, sr, low=200, high=3400)
    y_8k = bandpass_filter(x_8k, sr, low=200, high=3400)

    g_1k = _rms(y_1k) / _rms(x_1k)
    g_50 = _rms(y_50) / _rms(x_50)
    g_8k = _rms(y_8k) / _rms(x_8k)

    assert g_1k > 0.5
    assert g_50 < 0.1
    assert g_8k < 0.1


def test_denoise_reduces_noise_power_basic() -> None:
    sr = 16000
    seconds = 1.5
    noise_only_s = 0.3
    noise_only_n = int(sr * noise_only_s)
    clean_tone = _sine(sr, 1000.0, seconds - noise_only_s, amp=0.5)
    clean = ([0.0] * noise_only_n) + clean_tone

    rng = random.Random(0)
    noise = [rng.gauss(0.0, 0.2) for _ in range(len(clean))]
    noisy = [c + n for c, n in zip(clean, noise, strict=True)]

    denoised = denoise(noisy, sr, noise_estimate_ms=noise_only_s * 1000.0, threshold_mul=1.2, floor_gain=0.05)

    # Measure on the tone region only (the part we care about preserving).
    noisy_err = [n - c for n, c in zip(noisy[noise_only_n:], clean[noise_only_n:], strict=True)]
    denoised_err = [d - c for d, c in zip(denoised[noise_only_n:], clean[noise_only_n:], strict=True)]

    noisy_power = sum(e * e for e in noisy_err) / len(noisy_err)
    denoised_power = sum(e * e for e in denoised_err) / len(denoised_err)

    assert denoised_power < noisy_power * 0.9
