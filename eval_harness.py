# -*- coding: utf-8 -*-
"""
Phase 2 eval harness: synthesizes tone sequences from each raga's aroha/avroha/pakad
in config.RAAGA_DATABASE, runs them through the real pitch-extraction pipeline, and
scores classification accuracy. No microphone or external audio required.

Each (raga, sequence type) is rendered across several tonics and several noisy trials
(random per-note cents jitter, simulating a real singer's imprecision) rather than one
deterministic pure-tone pass, so accuracy reflects more than "can it match its own
noiseless reference histogram."

Usage: python3 eval_harness.py
"""

import json
import numpy as np

from config import SAMPLE_RATE, BUFFER_SIZE, SWARA_MAPPING, RAAGA_DATABASE
from feature_extraction import FeatureExtractor
from main_22 import RaagaClassifier

TONICS_HZ = [165.0, 220.0, 293.0]  # a few different vocal ranges, not just one Sa
NOTE_DURATION_S = 0.35              # per swara, long enough for several pitch-detection windows
TRIALS_PER_CASE = 3                 # repeated noisy renditions per (raga, sequence, tonic)
JITTER_CENTS = 15.0                 # max random per-note detune, simulating vocal imprecision
RANDOM_SEED = 42                    # fixed, so runs are comparable run-to-run

SEQUENCE_TYPES = ('aroha', 'avroha', 'pakad')


def swara_to_freq(swara_name, tonic_hz, cents_offset=0.0):
    semitone = SWARA_MAPPING[swara_name]
    cents = semitone * 100.0 + cents_offset
    return tonic_hz * (2.0 ** (cents / 1200.0))


def synthesize_note_sequence(swara_list, tonic_hz, rng, note_duration=NOTE_DURATION_S,
                              sample_rate=SAMPLE_RATE, jitter_cents=JITTER_CENTS):
    segments = []
    for swara in swara_list:
        jitter = rng.uniform(-jitter_cents, jitter_cents)
        freq = swara_to_freq(swara, tonic_hz, cents_offset=jitter)
        t = np.linspace(0, note_duration, int(sample_rate * note_duration), endpoint=False)
        segments.append(0.5 * np.sin(2 * np.pi * freq * t))
    return np.concatenate(segments) if segments else np.array([])


def build_histogram_from_audio(audio, tonic_hz, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE):
    extractor = FeatureExtractor(sample_rate=sample_rate, tonic_frequency=tonic_hz)
    for start in range(0, len(audio), buffer_size):
        chunk = audio[start:start + buffer_size]
        pitch = extractor.extract_pitch(chunk)
        if pitch is not None and np.isfinite(pitch):
            extractor.add_to_histogram(pitch)
    return extractor.get_normalized_histogram()


def run_eval():
    classifier = RaagaClassifier(RAAGA_DATABASE)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for raga_name, entry in RAAGA_DATABASE.items():
        for seq_type in SEQUENCE_TYPES:
            swara_list = entry[seq_type]
            for tonic_hz in TONICS_HZ:
                for trial in range(TRIALS_PER_CASE):
                    audio = synthesize_note_sequence(swara_list, tonic_hz, rng)
                    hist = build_histogram_from_audio(audio, tonic_hz)

                    if hist is None:
                        results.append({
                            'raga': raga_name, 'sequence_type': seq_type,
                            'tonic_hz': tonic_hz, 'trial': trial,
                            'predicted': None, 'correct': False, 'confidence': 0.0,
                        })
                        continue

                    scores = classifier.classify(hist)
                    predicted = max(scores, key=scores.get)
                    results.append({
                        'raga': raga_name, 'sequence_type': seq_type,
                        'tonic_hz': tonic_hz, 'trial': trial,
                        'predicted': predicted, 'correct': predicted == raga_name,
                        'confidence': round(float(scores[predicted]), 4),
                    })

    return results


def print_confusion_matrix(results):
    raga_names = list(RAAGA_DATABASE.keys())
    labels = raga_names + ['NONE']
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in raga_names}
    for r in results:
        predicted = r['predicted'] if r['predicted'] is not None else 'NONE'
        matrix[r['raga']][predicted] += 1

    col_w = 8
    print("\n Confusion matrix (rows = actual raga, cols = predicted):")
    header = " " * 10 + "".join(f"{name[:7]:>{col_w}}" for name in labels)
    print(header)
    for actual in raga_names:
        row = "".join(f"{matrix[actual][predicted]:>{col_w}}" for predicted in labels)
        print(f" {actual:<9}{row}")


def print_report(results):
    print("=" * 78)
    print(" PHASE 2 EVAL HARNESS - synthetic aroha/avroha/pakad classification")
    print("=" * 78)
    print(f" Tonics: {TONICS_HZ} Hz | Trials/case: {TRIALS_PER_CASE} | "
          f"Jitter: +/-{JITTER_CENTS} cents | Ragas: {len(RAAGA_DATABASE)}")
    print(f" Total test cases: {len(results)}")
    print("-" * 78)

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    print(f" Overall accuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)")

    print("\n Per-raga accuracy:")
    for raga_name in RAAGA_DATABASE:
        raga_results = [r for r in results if r['raga'] == raga_name]
        raga_correct = sum(1 for r in raga_results if r['correct'])
        pct = 100.0 * raga_correct / len(raga_results)
        print(f"   {raga_name:<10} {raga_correct}/{len(raga_results):<5} ({pct:.1f}%)")

    print("\n Per-sequence-type accuracy:")
    for seq_type in SEQUENCE_TYPES:
        seq_results = [r for r in results if r['sequence_type'] == seq_type]
        seq_correct = sum(1 for r in seq_results if r['correct'])
        pct = 100.0 * seq_correct / len(seq_results)
        print(f"   {seq_type:<10} {seq_correct}/{len(seq_results):<5} ({pct:.1f}%)")

    print("\n Per-tonic accuracy:")
    for tonic_hz in TONICS_HZ:
        tonic_results = [r for r in results if r['tonic_hz'] == tonic_hz]
        tonic_correct = sum(1 for r in tonic_results if r['correct'])
        pct = 100.0 * tonic_correct / len(tonic_results)
        print(f"   {tonic_hz:<10} {tonic_correct}/{len(tonic_results):<5} ({pct:.1f}%)")

    print_confusion_matrix(results)
    print("=" * 78)


if __name__ == '__main__':
    results = run_eval()
    print_report(results)
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull results written to eval_results.json")
