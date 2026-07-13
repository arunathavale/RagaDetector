# -*- coding: utf-8 -*-
"""
Phase 8 eval harness: same test matrix as eval_harness.py (every raga's
aroha/avroha/pakad, synthesized across several tonics and noisy trials), but
scored with SequenceClassifier (nyas sequence + aroha/avroha/pakad matching)
instead of RaagaClassifier (flat weighted histogram) - an apples-to-apples
comparison against the existing 95.2% baseline, using the exact same
synthetic audio generation so any accuracy difference is attributable to the
classification approach, not the test data.

Usage: python3 eval_sequence_harness.py
"""
import json
import numpy as np

from config import SAMPLE_RATE, BUFFER_SIZE, RAAGA_DATABASE
from feature_extraction import FeatureExtractor
from main_22 import SequenceClassifier, extract_nyas_sequence
from eval_harness import (
    TONICS_HZ, TRIALS_PER_CASE, JITTER_CENTS, RANDOM_SEED, SEQUENCE_TYPES,
    synthesize_note_sequence, print_confusion_matrix, print_report,
)


def build_nyas_sequence_from_audio(audio, tonic_hz, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE):
    extractor = FeatureExtractor(sample_rate=sample_rate, tonic_frequency=tonic_hz)
    chunks = [audio[i:i + buffer_size] for i in range(0, len(audio), buffer_size)]
    pitches = []
    for chunk in chunks:
        pitch = extractor.extract_pitch(chunk)
        if pitch is not None and np.isfinite(pitch):
            pitches.append(pitch)
    return extract_nyas_sequence(extractor, pitches)


def run_eval():
    classifier = SequenceClassifier(RAAGA_DATABASE)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for raga_name, entry in RAAGA_DATABASE.items():
        for seq_type in SEQUENCE_TYPES:
            swara_list = entry[seq_type]
            for tonic_hz in TONICS_HZ:
                for trial in range(TRIALS_PER_CASE):
                    audio = synthesize_note_sequence(swara_list, tonic_hz, rng)
                    nyas_sequence = build_nyas_sequence_from_audio(audio, tonic_hz)

                    if not nyas_sequence:
                        results.append({
                            'raga': raga_name, 'sequence_type': seq_type,
                            'tonic_hz': tonic_hz, 'trial': trial,
                            'predicted': None, 'correct': False, 'confidence': 0.0,
                        })
                        continue

                    scores = classifier.classify(nyas_sequence)
                    predicted = max(scores, key=scores.get)
                    results.append({
                        'raga': raga_name, 'sequence_type': seq_type,
                        'tonic_hz': tonic_hz, 'trial': trial,
                        'predicted': predicted, 'correct': predicted == raga_name,
                        'confidence': round(float(scores[predicted]), 4),
                    })

    return results


if __name__ == '__main__':
    print("=" * 78)
    print(" PHASE 8 EVAL HARNESS - sequence (nyas + aroha/avroha/pakad) classification")
    print("=" * 78)
    print(f" Tonics: {TONICS_HZ} Hz | Trials/case: {TRIALS_PER_CASE} | Jitter: +/-{JITTER_CENTS} cents | Ragas: {len(RAAGA_DATABASE)}")
    results = run_eval()
    print_report(results)
    with open('eval_sequence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull results written to eval_sequence_results.json")
