# -*- coding: utf-8 -*-
"""
Find the tonic that minimizes deviation from a target raga's reference histogram,
by exploiting the fact that changing tonic is mathematically equivalent to a
circular shift of the pitch-class histogram (cents = 1200*log2(f/tonic) mod 1200).

Detects pitches ONCE (pitch detection doesn't depend on tonic at all), then tests
all 120 possible 10-cent rotations via cheap array shifts - exact, not a heuristic,
and far faster than re-running pitch detection per candidate tonic.

Usage:
  python3 find_optimal_tonic.py <file.wav> <intended_raga> [min_hz] [max_hz]
  python3 find_optimal_tonic.py <file.wav> --blind [min_hz] [max_hz]

--blind mode doesn't assume you already know the raga: it searches every (tonic,
raga) combination simultaneously and reports whichever pairing gives the single
strongest, most confident match. Intuition: a correct tonic applied to genuine
singing of some real raga should make that raga's reference win decisively; wrong
tonics tend to produce mediocre, ambiguous matches against everything.
"""
import sys
import numpy as np

from config import SAMPLE_RATE, HISTOGRAM_BINS, RAAGA_DATABASE
from feature_extraction import FeatureExtractor
from main_22 import load_file_chunks, TONIC_DETECTION_SECONDS, BUFFER_SIZE, RaagaClassifier

CENTS_PER_BIN = 1200.0 / HISTOGRAM_BINS


def detect_raw_pitches(chunks, method='autocorrelation'):
    """Detect pitches once - tonic-independent, so this only needs to run once
    regardless of how many candidate tonics get evaluated afterward."""
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    pitches = []
    for c in chunks:
        p = extractor.extract_pitch(c, method=method)
        if p and not np.isnan(p):
            pitches.append(p)
    return pitches


def build_anchor_histogram(pitches, anchor_hz=1.0):
    """Histogram built with an arbitrary fixed reference tonic - equivalent to
    tracking each pitch's absolute log-position mod one octave. Every other
    tonic's histogram is just a circular shift of this one."""
    hist = np.zeros(HISTOGRAM_BINS)
    for p in pitches:
        cents = (1200.0 * np.log2(p / anchor_hz)) % 1200.0
        b = int(cents / CENTS_PER_BIN) % HISTOGRAM_BINS
        hist[b] += 1
    total = np.sum(hist)
    return hist / total if total > 0 else hist


def shift_for_tonic(tonic_hz, anchor_hz=1.0):
    """Bin shift equivalent to a given tonic, relative to the anchor histogram."""
    cents = (1200.0 * np.log2(tonic_hz / anchor_hz)) % 1200.0
    return int(round(cents / CENTS_PER_BIN)) % HISTOGRAM_BINS


def histogram_for_tonic(anchor_hist, tonic_hz, anchor_hz=1.0):
    shift = shift_for_tonic(tonic_hz, anchor_hz)
    return np.roll(anchor_hist, -shift)


def find_best_tonic(anchor_hist, intended_raga, search_min_hz, search_max_hz, anchor_hz=1.0):
    """Scan every semitone-bin shift (120 per octave) across every octave that
    falls within [search_min_hz, search_max_hz], scoring each shifted histogram
    through the EXACT SAME RaagaClassifier.classify() the real pipeline uses
    (including its per-swara noise threshold) - not raw cosine similarity, which
    scores differently since it skips that thresholding step entirely.

    Ranked by GAP over the 2nd-best raga, not raw score: classify() min-max
    normalizes across all 7 ragas, so whichever raga wins always shows exactly
    100% regardless of how confidently it won - raw score can't distinguish a
    landslide from a coin-flip, only the gap over the runner-up can."""
    classifier = RaagaClassifier(RAAGA_DATABASE)
    results = []
    import math
    lowest_octave_hz = anchor_hz * (2 ** math.floor(math.log2(search_min_hz / anchor_hz)))
    hz = lowest_octave_hz
    while hz <= search_max_hz * 2:
        for shift in range(HISTOGRAM_BINS):
            tonic_hz = hz * (2 ** (shift * CENTS_PER_BIN / 1200.0))
            if search_min_hz <= tonic_hz <= search_max_hz:
                hist = histogram_for_tonic(anchor_hist, tonic_hz, anchor_hz)
                scores = classifier.classify(hist)
                winner = max(scores, key=scores.get)
                others = sorted([s for r, s in scores.items() if r != winner], reverse=True)
                gap = scores[winner] - (others[0] if others else 0.0)
                signed_gap = gap if winner == intended_raga else -gap
                results.append((tonic_hz, scores[intended_raga], signed_gap, scores))
        hz *= 2
    results.sort(key=lambda x: -x[2])
    return results


def find_best_tonic_and_raga(anchor_hist, search_min_hz, search_max_hz, anchor_hz=1.0):
    """Blind search: find the (tonic, raga) combination that gives the single
    strongest, most confident match across ALL 7 ragas simultaneously - doesn't
    require already knowing which raga is being performed."""
    classifier = RaagaClassifier(RAAGA_DATABASE)
    results = []
    import math
    lowest_octave_hz = anchor_hz * (2 ** math.floor(math.log2(search_min_hz / anchor_hz)))
    hz = lowest_octave_hz
    while hz <= search_max_hz * 2:
        for shift in range(HISTOGRAM_BINS):
            tonic_hz = hz * (2 ** (shift * CENTS_PER_BIN / 1200.0))
            if search_min_hz <= tonic_hz <= search_max_hz:
                hist = histogram_for_tonic(anchor_hist, tonic_hz, anchor_hz)
                scores = classifier.classify(hist)
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                winner, winner_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                gap = winner_score - second_score
                results.append((tonic_hz, winner, gap, scores))
        hz *= 2
    results.sort(key=lambda x: -x[2])
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 find_optimal_tonic.py <file.wav> <intended_raga> [min_hz] [max_hz]")
        print("  python3 find_optimal_tonic.py <file.wav> --blind [min_hz] [max_hz]")
        sys.exit(1)
    filepath = sys.argv[1]
    blind = sys.argv[2] == '--blind'
    intended_raga = None if blind else sys.argv[2]
    search_min = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    search_max = float(sys.argv[4]) if len(sys.argv) > 4 else 500.0

    chunks = load_file_chunks(filepath)
    detect_chunk_count = int(TONIC_DETECTION_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    performance_chunks = chunks[detect_chunk_count:] or chunks

    print(f"Detecting pitches once across {len(performance_chunks)} chunks...")
    pitches = detect_raw_pitches(performance_chunks)
    print(f"Got {len(pitches)} raw pitch detections.\n")

    anchor_hist = build_anchor_histogram(pitches)

    if blind:
        results = find_best_tonic_and_raga(anchor_hist, search_min, search_max)
        print(f"Top 10 (tonic, raga) candidates in [{search_min}, {search_max}] Hz - "
              f"blind search, no raga assumed:")
        print(f"{'Tonic Hz':<12} {'Best raga':<10} {'gap over 2nd':<12}")
        for tonic_hz, winner, gap, scores in results[:10]:
            print(f"{tonic_hz:<12.2f} {winner:<10} {gap*100:+.1f}%")
    else:
        results = find_best_tonic(anchor_hist, intended_raga, search_min, search_max)
        print(f"Top 10 candidate tonics in [{search_min}, {search_max}] Hz "
              f"(exact search over all {HISTOGRAM_BINS} bin-shifts per octave, "
              f"scored via the real classifier, ranked by gap over 2nd place):")
        print(f"{'Tonic Hz':<12} {'Winner':<10} {'2nd best':<20} {'signed gap':<10}")
        for tonic_hz, score, signed_gap, scores in results[:10]:
            winner = max(scores, key=scores.get)
            others = sorted([(r, s) for r, s in scores.items() if r != winner], key=lambda x: -x[1])
            second = others[0]
            print(f"{tonic_hz:<12.2f} {winner:<10} {second[0]+'='+str(round(second[1]*100,1))+'%':<20} {signed_gap*100:+.1f}%")


if __name__ == '__main__':
    main()
