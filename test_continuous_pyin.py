# -*- coding: utf-8 -*-
"""
Re-test pyin fairly: the earlier comparison (compare_pitch_methods.py) ran pyin on
isolated ~23ms chunks with zero temporal continuity between them, which defeats the
Viterbi decoding pyin actually relies on to disambiguate a moving melody from a
stable drone. This runs pyin ONCE, continuously, across the whole file with a
proper ~10ms hop - the way it's actually designed to be used - and compares against
the existing chunked-autocorrelation result for the same file and tonic.

Usage: python3 test_continuous_pyin.py <file.wav> <tonic_hz> <intended_raga>
"""
import sys
import numpy as np
import librosa

from config import SAMPLE_RATE, HISTOGRAM_BINS, BINS_PER_SEMITONE, RAAGA_DATABASE
from main_22 import RaagaClassifier, compute_swara_deviation_report, TONIC_DETECTION_SECONDS

CENTS_PER_BIN = 1200.0 / HISTOGRAM_BINS
HOP_LENGTH = int(SAMPLE_RATE * 0.01)  # ~10ms, matching SWARA_EXTRACTION_SPEC.md's
# recommendation - computed from SAMPLE_RATE rather than hardcoded


def pitch_to_bin(pitch_hz, tonic_hz):
    cents = (1200.0 * np.log2(pitch_hz / tonic_hz)) % 1200.0
    return int(cents / CENTS_PER_BIN) % HISTOGRAM_BINS


def run_continuous_pyin(audio, tonic_hz, voiced_prob_threshold=0.5):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=audio,
        fmin=80.0,
        fmax=1200.0,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    valid = voiced_flag & (voiced_probs >= voiced_prob_threshold) & np.isfinite(f0)
    total_frames = len(f0)
    valid_pitches = f0[valid]

    hist = np.zeros(HISTOGRAM_BINS)
    for p in valid_pitches:
        hist[pitch_to_bin(p, tonic_hz)] += 1
    total = np.sum(hist)
    hist = hist / total if total > 0 else hist

    return hist, len(valid_pitches), total_frames


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 test_continuous_pyin.py <file.wav> <tonic_hz> <intended_raga>")
        sys.exit(1)
    filepath, tonic_hz, intended_raga = sys.argv[1], float(sys.argv[2]), sys.argv[3]

    print(f"Loading {filepath} as a continuous signal (not chunked)...")
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    # Skip the same opening calibration window as the rest of the pipeline does.
    skip_samples = int(TONIC_DETECTION_SECONDS * SAMPLE_RATE)
    performance_audio = audio[skip_samples:] if len(audio) > skip_samples else audio

    print(f"Running continuous pyin (hop={HOP_LENGTH} samples, ~{HOP_LENGTH/SAMPLE_RATE*1000:.1f}ms) "
          f"over {len(performance_audio)/SAMPLE_RATE:.1f}s of audio - this may take a while...")
    hist, n_valid, n_total = run_continuous_pyin(performance_audio, tonic_hz)

    print(f"\nValid voiced+confident frames: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    classifier = RaagaClassifier(RAAGA_DATABASE)
    scores = classifier.classify(hist)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print("\nTop 3:", ", ".join(f"{r}={s*100:.1f}%" for r, s in ranked[:3]))
    print(f"Detected raga: {ranked[0][0]}")

    report = compute_swara_deviation_report(hist, intended_raga)
    print("\nBiggest deviations from target:")
    for r in report[:4]:
        sign = "+" if r["deviation_pct"] >= 0 else ""
        print(f"  {r['swara']:<12} actual {r['actual_pct']:>5.1f}% | target {r['target_pct']:>5.1f}% | "
              f"deviation {sign}{r['deviation_pct']:>5.1f}%")


if __name__ == '__main__':
    main()
