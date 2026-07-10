# -*- coding: utf-8 -*-
"""
Compare pitch-detection methods (autocorrelation vs yin) on a real saved recording,
to see which tracks the vocal melody more robustly through Tabla/Tanpura/Harmonium
accompaniment. Reuses the exact same file-mode pipeline (load -> per-chunk pitch
extract -> one-hot frame -> full-session histogram -> classify), just swapping the
pitch-detection method, so it's an apples-to-apples comparison - same audio, same
tonic, only the pitch algorithm differs.

Usage: python3 compare_pitch_methods.py <recording.wav> <tonic_hz> <intended_raga>
"""
import sys
import numpy as np

from config import SAMPLE_RATE, BUFFER_SIZE, RAAGA_DATABASE
from feature_extraction import FeatureExtractor
from main_22 import (
    RaagaClassifier, load_file_chunks, pitch_to_frame_histogram,
    compute_swara_deviation_report, subtract_swara_noise_floor, TONIC_DETECTION_SECONDS,
)

METHODS = ('autocorrelation', 'yin', 'pyin')


def analyze_with_method(chunks, tonic_hz, method):
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, tonic_frequency=tonic_hz)
    history = []
    detections = 0
    attempts = 0
    for chunk in chunks:
        attempts += 1
        p = extractor.extract_pitch(chunk, method=method)
        if p and not np.isnan(p):
            detections += 1
            frame_hist = pitch_to_frame_histogram(extractor, p)
            if frame_hist is not None:
                history.append(frame_hist)
    if not history:
        return None
    ch = np.mean(history, axis=0)
    ch /= np.sum(ch) if np.sum(ch) > 0 else 1
    return {
        'histogram': ch,
        'detections': detections,
        'attempts': attempts,
    }


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 compare_pitch_methods.py <recording.wav> <tonic_hz> <intended_raga>")
        sys.exit(1)
    filepath, tonic_hz, intended_raga = sys.argv[1], float(sys.argv[2]), sys.argv[3]

    chunks = load_file_chunks(filepath)
    # Skip the opening calibration segment, same as run_file_mode() does - it's a
    # held-Sa reference, not performance content.
    detect_chunk_count = int(TONIC_DETECTION_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    performance_chunks = chunks[detect_chunk_count:] or chunks

    classifier = RaagaClassifier(RAAGA_DATABASE)

    print("=" * 70)
    print(f" PITCH-METHOD COMPARISON: {filepath}")
    print(f" Tonic: {tonic_hz} Hz | Intended raga: {intended_raga} | Chunks: {len(performance_chunks)}")
    print("=" * 70)

    def report_variant(label, hist):
        print(f"\n--- {label} ---")
        scores = classifier.classify(hist)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(" Top 3:", ", ".join(f"{r}={s*100:.1f}%" for r, s in top3))
        print(f" Detected raga: {max(scores, key=scores.get)}")
        report = compute_swara_deviation_report(hist, intended_raga)
        print(" Biggest deviations from target:")
        for r in report[:4]:
            sign = "+" if r["deviation_pct"] >= 0 else ""
            print(f"   {r['swara']:<12} actual {r['actual_pct']:>5.1f}% | target {r['target_pct']:>5.1f}% | "
                  f"deviation {sign}{r['deviation_pct']:>5.1f}%")

    results = {}
    for method in METHODS:
        result = analyze_with_method(performance_chunks, tonic_hz, method)
        results[method] = result
        print(f"\n=== Method: {method} ===")
        if result is None:
            print(" No pitch detected at all with this method.")
            continue
        pct = 100.0 * result['detections'] / result['attempts'] if result['attempts'] else 0.0
        print(f" Detection rate: {result['detections']}/{result['attempts']} ({pct:.1f}%)")

        report_variant("raw histogram", result['histogram'])
        floored = subtract_swara_noise_floor(result['histogram'])
        report_variant("noise-floor subtracted", floored)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
