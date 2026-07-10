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
  python3 find_optimal_tonic.py <file.wav> --drone [min_hz] [max_hz]

--blind mode doesn't assume you already know the raga: it searches every (tonic,
raga) combination simultaneously and reports whichever pairing gives the single
strongest, most confident match. Intuition: a correct tonic applied to genuine
singing of some real raga should make that raga's reference win decisively; wrong
tonics tend to produce mediocre, ambiguous matches against everything. Tested and
found NOT fully reliable on its own - a coincidentally strong wrong (tonic, raga)
pairing can outrank the true one on noisy real recordings.

--drone mode doesn't need a raga OR the classifier at all: it scores candidates by
whether a tanpura-style drone dyad (Sa-Pa, a fifth, or Sa-Ma, a fourth) is present
alongside the candidate Sa peak - the tanpura almost always drones on one of those,
so this is a raga-agnostic structural signal, independent of (and complementary
to) the raga-matching searches above.

Every run also reports a within-recording stability check: the file is split into
several segments and each is independently searched for its best drone-prior
tonic. Since tonic essentially never changes mid-performance, segments that
disagree point to a pitch-tracking problem in that portion of the recording, not
an actual tonic shift.
"""
import sys
import numpy as np

from config import SAMPLE_RATE, HISTOGRAM_BINS, BINS_PER_SEMITONE, RAAGA_DATABASE
from feature_extraction import FeatureExtractor
from main_22 import load_file_chunks, TONIC_DETECTION_SECONDS, BUFFER_SIZE, RaagaClassifier

CENTS_PER_BIN = 1200.0 / HISTOGRAM_BINS
FOURTH_SEMITONE = 5   # Ma_shuddha - Sa-Ma drone dyad
FIFTH_SEMITONE = 7    # Pa - Sa-Pa drone dyad, the more common Hindustani tuning


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


def score_drone_priors(anchor_hist, tonic_hz, anchor_hz=1.0, fourth_weight=0.5, fifth_weight=0.8):
    """Score a candidate tonic by drone-interval priors, independent of any raga
    or classifier: a tanpura almost always drones on Sa+Pa (a fifth, 700 cents) or
    Sa+Ma (a fourth, 500 cents), so a genuinely correct tonic candidate should show
    a strong peak at its own position AND at one of those two intervals - not just
    a strong peak on its own, which any of the 12 semitones could show by chance.
    Sa-Pa is the more common Hindustani drone tuning, hence the higher weight."""
    shift = shift_for_tonic(tonic_hz, anchor_hz)
    hist = np.roll(anchor_hist, -shift)
    sa_strength = np.sum(hist[0:BINS_PER_SEMITONE])
    ma_strength = np.sum(hist[FOURTH_SEMITONE*BINS_PER_SEMITONE:(FOURTH_SEMITONE+1)*BINS_PER_SEMITONE])
    pa_strength = np.sum(hist[FIFTH_SEMITONE*BINS_PER_SEMITONE:(FIFTH_SEMITONE+1)*BINS_PER_SEMITONE])
    drone_bonus = max(fourth_weight * ma_strength, fifth_weight * pa_strength)
    return float(sa_strength + drone_bonus), float(sa_strength), float(ma_strength), float(pa_strength)


def find_best_tonic_by_drone_priors(anchor_hist, search_min_hz, search_max_hz, anchor_hz=1.0):
    """Raga-agnostic tonic search: doesn't need a target raga or the classifier at
    all (the blind classifier search was tested and found unreliable on its own -
    see module docstring) - just looks for a candidate Sa position with a strong
    drone dyad (Sa-Pa or Sa-Ma), the tanpura's actual structural signature."""
    results = []
    import math
    lowest_octave_hz = anchor_hz * (2 ** math.floor(math.log2(search_min_hz / anchor_hz)))
    hz = lowest_octave_hz
    while hz <= search_max_hz * 2:
        for shift in range(HISTOGRAM_BINS):
            tonic_hz = hz * (2 ** (shift * CENTS_PER_BIN / 1200.0))
            if search_min_hz <= tonic_hz <= search_max_hz:
                score, sa, ma, pa = score_drone_priors(anchor_hist, tonic_hz, anchor_hz)
                results.append((tonic_hz, score, sa, ma, pa))
        hz *= 2
    results.sort(key=lambda x: -x[1])
    return results


def check_tonic_stability(chunks, search_min_hz, search_max_hz, n_segments=4, anchor_hz=1.0):
    """Split the recording into N roughly-equal segments and independently find
    the best drone-prior tonic candidate in each. Tonic essentially never changes
    mid-performance in Hindustani classical music - segments whose estimates
    disagree point to a pitch-tracking/audio-quality problem in that portion of
    the recording, not a genuine tonic shift."""
    seg_len = len(chunks) // n_segments
    if seg_len == 0:
        return []
    results = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else len(chunks)
        pitches = detect_raw_pitches(chunks[start:end])
        if not pitches:
            results.append((i, None, 0))
            continue
        seg_anchor_hist = build_anchor_histogram(pitches)
        drone_results = find_best_tonic_by_drone_priors(seg_anchor_hist, search_min_hz, search_max_hz, anchor_hz)
        top_tonic = drone_results[0][0] if drone_results else None
        results.append((i, top_tonic, len(pitches)))
    return results


def print_stability_check(chunks, search_min, search_max):
    print("\n" + "-" * 70)
    print(" WITHIN-RECORDING STABILITY CHECK (drone-prior tonic per segment):")
    print(" Tonic essentially never changes mid-performance - segments that")
    print(" disagree point to a pitch-tracking problem in that portion, not a")
    print(" real tonic shift.")
    print("-" * 70)
    stability = check_tonic_stability(chunks, search_min, search_max)
    for i, tonic_hz, n_pitches in stability:
        if tonic_hz is None:
            print(f"  Segment {i+1}: no pitch detections")
        else:
            print(f"  Segment {i+1}: {tonic_hz:.2f} Hz  ({n_pitches} detections)")
    valid = [t for _, t, _ in stability if t is not None]
    if len(valid) >= 2:
        # Compare in cents (octave-folded) so e.g. 103 Hz and 207 Hz read as agreeing.
        ref = valid[0]
        diffs_cents = [abs(((1200 * np.log2(t / ref) + 600) % 1200) - 600) for t in valid[1:]]
        max_diff = max(diffs_cents) if diffs_cents else 0.0
        verdict = "consistent" if max_diff < 50 else "DISAGREE - investigate that segment"
        print(f"  Max spread across segments: {max_diff:.0f} cents ({verdict})")
    print("-" * 70)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 find_optimal_tonic.py <file.wav> <intended_raga> [min_hz] [max_hz]")
        print("  python3 find_optimal_tonic.py <file.wav> --blind [min_hz] [max_hz]")
        print("  python3 find_optimal_tonic.py <file.wav> --drone [min_hz] [max_hz]")
        sys.exit(1)
    filepath = sys.argv[1]
    mode = sys.argv[2] if sys.argv[2] in ('--blind', '--drone') else None
    intended_raga = None if mode else sys.argv[2]
    search_min = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    search_max = float(sys.argv[4]) if len(sys.argv) > 4 else 500.0

    chunks = load_file_chunks(filepath)
    detect_chunk_count = int(TONIC_DETECTION_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    performance_chunks = chunks[detect_chunk_count:] or chunks

    print(f"Detecting pitches once across {len(performance_chunks)} chunks...")
    pitches = detect_raw_pitches(performance_chunks)
    print(f"Got {len(pitches)} raw pitch detections.\n")

    anchor_hist = build_anchor_histogram(pitches)

    if mode == '--blind':
        results = find_best_tonic_and_raga(anchor_hist, search_min, search_max)
        print(f"Top 10 (tonic, raga) candidates in [{search_min}, {search_max}] Hz - "
              f"blind search, no raga assumed:")
        print(f"{'Tonic Hz':<12} {'Best raga':<10} {'gap over 2nd':<12}")
        for tonic_hz, winner, gap, scores in results[:10]:
            print(f"{tonic_hz:<12.2f} {winner:<10} {gap*100:+.1f}%")
    elif mode == '--drone':
        results = find_best_tonic_by_drone_priors(anchor_hist, search_min, search_max)
        print(f"Top 10 candidate tonics in [{search_min}, {search_max}] Hz - "
              f"drone-interval priors (Sa-Pa/Sa-Ma dyad strength), no raga or "
              f"classifier involved:")
        print(f"{'Tonic Hz':<12} {'Score':<8} {'Sa':<8} {'Ma(4th)':<9} {'Pa(5th)':<9}")
        for tonic_hz, score, sa, ma, pa in results[:10]:
            print(f"{tonic_hz:<12.2f} {score*100:<7.1f}% {sa*100:<7.1f}% {ma*100:<8.1f}% {pa*100:<8.1f}%")
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

    print_stability_check(performance_chunks, search_min, search_max)


if __name__ == '__main__':
    main()
