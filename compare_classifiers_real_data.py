# -*- coding: utf-8 -*-
"""
Side-by-side comparison of RaagaClassifier (flat histogram) vs SequenceClassifier
(nyas sequence + aroha/avroha/pakad matching) on real recordings with
independently-validated tonics - the actual point of building SequenceClassifier
was to fix real failures the histogram approach hit this session (Kaushiki
Chakraborty's meend contamination, bhoop03.wav's Bhupali/Yaman near-tie), so
this checks whether it actually helps on those specific cases, not just the
synthetic eval harness.

Usage: python3 compare_classifiers_real_data.py
"""
import numpy as np

from main_22 import (
    load_file_chunks, FeatureExtractor, RaagaClassifier, SequenceClassifier,
    RAGA_REGISTRY, pitch_to_frame_histogram, extract_nyas_sequence,
    SAMPLE_RATE,
)

# (filepath, intended_raga, validated_tonic_hz, note)
TEST_CASES = [
    ("recordings/Yaman_Kaushiki_20260711_183007.wav", "Yaman", 110.15,
     "octave-validated against her documented 207.6-220 Hz range"),
    ("recordings/bhoop03.wav", "Bhupali", 171.85,
     "best-found tonic, only a 1.2-point histogram win over Yaman"),
    ("reference_audio/Yaman_BhimsenJoshi_EliAliPiyaBin.mp3", "Yaman", 110.79,
     "classifier-validated, this project's primary reference file"),
    ("recordings/Bhairav_20260709_203955.wav", "Bhairav", 137.19,
     "validated against Bhimsen Joshi's documented 135-138.6 Hz range"),
    ("recordings/Bhairav_20260709_211312.wav", "Bhairav", 132.51,
     "validated tonic from this morning's analyze_recordings.py run"),
]


def top3(scores):
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return ", ".join(f"{r}={s*100:.1f}%" for r, s in ranked)


def main():
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    hist_classifier = RaagaClassifier(RAGA_REGISTRY)
    seq_classifier = SequenceClassifier(RAGA_REGISTRY)

    for filepath, intended_raga, tonic_hz, note in TEST_CASES:
        print(f"\n{'='*90}\n{filepath}  (label: {intended_raga}, tonic {tonic_hz} Hz)\n  {note}\n{'-'*90}")
        chunks = load_file_chunks(filepath)
        extractor.set_tonic(tonic_hz)

        pitches = []
        frames = []
        for c in chunks:
            p = extractor.extract_pitch(c)
            if p and not np.isnan(p):
                pitches.append(p)
                fh = pitch_to_frame_histogram(extractor, p)
                if fh is not None:
                    frames.append(fh)

        if not frames:
            print("  No pitch detected at all.")
            continue

        hist = np.mean(frames, axis=0)
        hist /= np.sum(hist) if np.sum(hist) > 0 else 1
        hist_scores = hist_classifier.classify(hist)
        hist_winner = max(hist_scores, key=hist_scores.get)

        nyas_sequence = extract_nyas_sequence(extractor, pitches)
        seq_scores = seq_classifier.classify(nyas_sequence)
        seq_winner = max(seq_scores, key=seq_scores.get)

        print(f"  Histogram : {hist_winner:<10} {'(correct)' if hist_winner == intended_raga else '(WRONG)':<10} | {top3(hist_scores)}")
        print(f"  Sequence  : {seq_winner:<10} {'(correct)' if seq_winner == intended_raga else '(WRONG)':<10} | {top3(seq_scores)}")
        print(f"  ({len(nyas_sequence)} nyas events extracted from {len(pitches)} raw pitch detections)")


if __name__ == "__main__":
    main()
