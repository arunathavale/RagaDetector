# -*- coding: utf-8 -*-
"""
Phase A tonic-detection comparison: for each saved recording, compute several
candidate tonic-finding methods over the same first LIVE_TONIC_WINDOW_SECONDS
of audio and tabulate them side by side, so we can see which method (if any)
is worth wiring into run_live_mode() in place of the current raw drone-prior
search - rather than guessing.

Methods compared:
  drone_raw    - current run_live_mode() method: drone-interval priors (Sa-Pa/
                 Sa-Ma dyad strength) over ALL detected pitches in the window.
  drone_stable - same drone-prior search, but restricted to pitches that pass
                 filter_stable_pitches() first (frame-to-frame steady, i.e.
                 plausibly a held tone rather than mid-glide).
  median       - the older, simpler auto_detect_tonic(method='median') idea:
                 straight median of all detected pitches in the window, no
                 octave-search or drone reasoning at all.
  hpss         - drone-prior search, but pitch detection runs on the HARMONIC
                 component of the tonic window after librosa's harmonic-
                 percussive source separation, instead of the raw signal.
                 Tabla hits/consonant transients are percussive; the tanpura
                 drone and sung notes are harmonic - stripping the percussive
                 layer first should reduce a different kind of contamination
                 than filter_stable_pitches() targets (that one screens for
                 gliding vs. steady pitch, not for what KIND of sound produced
                 a steady pitch - a steady tabla resonance passes it just as
                 easily as a steady drone or held vocal note).
  pyin         - drone-prior search using librosa.pyin() run continuously
                 across the whole tonic window (proper hop length, real
                 temporal context for its Viterbi decoding), instead of the
                 default per-chunk autocorrelation. Earlier this session pyin
                 was ruled out for LIVE swara tracking specifically because
                 it's source-agnostic (confident about any stable pitch, not
                 just the voice) - a liability there, but plausibly an asset
                 for isolating a stable drone specifically.
  targeted*    - DIAGNOSTIC ONLY. Exploits the filename's raga label to search
                 for the tonic that makes that specific raga classify most
                 confidently (the technique this project's live pipeline
                 explicitly moved away from, since a real deployment doesn't
                 know the raga in advance - see main_22.py's
                 determine_tonic_from_pitches() docstring). Included here
                 purely as a labeled-data reference point to gauge how far off
                 the raga-agnostic methods are on each file - never wire this
                 into anything live.

Usage: python3 compare_tonic_methods.py
"""
import os
import glob
import numpy as np
import librosa

from main_22 import (
    load_file_chunks, FeatureExtractor, RaagaClassifier, RAGA_REGISTRY,
    determine_tonic_from_pitches, filter_stable_pitches, continuous_pyin_pitches,
    HISTOGRAM_BINS, TYPICAL_TONIC_RANGE_HZ,
    SAMPLE_RATE, BUFFER_SIZE, LIVE_TONIC_WINDOW_SECONDS, RECORDINGS_DIR,
)


def tonic_targeted_diagnostic(pitches, intended_raga, classifier,
                               search_min_hz=TYPICAL_TONIC_RANGE_HZ[0],
                               search_max_hz=TYPICAL_TONIC_RANGE_HZ[1]):
    """DIAGNOSTIC ONLY - see module docstring. Not used by any live code path."""
    if not pitches:
        return None, None
    cents_per_bin = 1200.0 / HISTOGRAM_BINS
    anchor_hz = 1.0
    anchor_hist = np.zeros(HISTOGRAM_BINS)
    for p in pitches:
        cents = (1200.0 * np.log2(p / anchor_hz)) % 1200.0
        b = int(cents / cents_per_bin) % HISTOGRAM_BINS
        anchor_hist[b] += 1
    total = np.sum(anchor_hist)
    if total == 0:
        return None, None
    anchor_hist /= total

    lowest_octave_hz = anchor_hz * (2 ** np.floor(np.log2(search_min_hz / anchor_hz)))
    best_tonic, best_gap = None, -2.0
    hz = lowest_octave_hz
    while hz <= search_max_hz * 2:
        for shift in range(HISTOGRAM_BINS):
            tonic_hz = hz * (2 ** (shift * cents_per_bin / 1200.0))
            if search_min_hz <= tonic_hz <= search_max_hz:
                shifted = np.roll(anchor_hist, -shift)
                scores = classifier.classify(shifted)
                winner = max(scores, key=scores.get)
                others = sorted((s for r, s in scores.items() if r != winner), reverse=True)
                gap = scores[winner] - (others[0] if others else 0.0)
                signed_gap = gap if winner == intended_raga else -gap
                if signed_gap > best_gap:
                    best_gap = signed_gap
                    best_tonic = tonic_hz
        hz *= 2
    return best_tonic, best_gap


def hpss_harmonic_pitches(raw_audio, extractor, buffer_size=BUFFER_SIZE):
    if len(raw_audio) == 0:
        return []
    harmonic, _ = librosa.effects.hpss(raw_audio)
    pitches = []
    for i in range(0, len(harmonic), buffer_size):
        chunk = harmonic[i:i + buffer_size]
        if len(chunk) < 2:
            continue
        p = extractor.extract_pitch(chunk)
        if p and not np.isnan(p):
            pitches.append(p)
    return pitches


def stability_spread(pitch_source_fn, pitches, f_tonic, n_segments=4):
    """pitch_source_fn: takes a pitch list, returns a tonic_hz (first return value
    matters) - lets this work for both drone_raw and drone_stable by passing a
    different tonic-finding call per segment."""
    if len(pitches) < n_segments * 10:
        return None
    seg_len = len(pitches) // n_segments
    offsets = []
    for i in range(n_segments):
        seg = pitches[i*seg_len:] if i == n_segments - 1 else pitches[i*seg_len:(i+1)*seg_len]
        seg_tonic = pitch_source_fn(seg)
        if seg_tonic is not None:
            offsets.append(1200.0 * np.log2(seg_tonic / f_tonic))
    return round(max(offsets) - min(offsets), 0) if offsets else None


def analyze_one(filepath, extractor, classifier):
    filename = os.path.basename(filepath)
    intended_raga = filename.split('_')[0]  # diagnostic label only

    chunks = load_file_chunks(filepath)
    n_tonic_chunks = int(LIVE_TONIC_WINDOW_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    tonic_chunks = chunks[:n_tonic_chunks]

    pitches = []
    for c in tonic_chunks:
        p = extractor.extract_pitch(c)
        if p and not np.isnan(p):
            pitches.append(p)

    if not pitches:
        return {"file": filename, "intended_raga": intended_raga, "error": "no pitch detected"}

    stable_pitches = filter_stable_pitches(pitches)
    raw_audio = np.concatenate(tonic_chunks).astype(np.float32) if tonic_chunks else np.array([])
    hpss_pitches = hpss_harmonic_pitches(raw_audio, extractor)
    pyin_pitches = continuous_pyin_pitches(raw_audio)

    pyin_stable_pitches = filter_stable_pitches(pyin_pitches) if pyin_pitches else []

    drone_raw, *_ = determine_tonic_from_pitches(pitches)
    drone_stable, *_ = determine_tonic_from_pitches(stable_pitches) if stable_pitches else (None,)
    drone_hpss, *_ = determine_tonic_from_pitches(hpss_pitches) if hpss_pitches else (None,)
    drone_pyin, *_ = determine_tonic_from_pitches(pyin_pitches) if pyin_pitches else (None,)
    drone_pyin_stable, *_ = determine_tonic_from_pitches(pyin_stable_pitches) if pyin_stable_pitches else (None,)
    median_tonic = float(np.median(pitches))
    targeted, targeted_gap = tonic_targeted_diagnostic(pitches, intended_raga, classifier)

    drone_raw_spread = stability_spread(lambda seg: determine_tonic_from_pitches(seg)[0], pitches, drone_raw) \
        if drone_raw else None
    drone_stable_spread = stability_spread(
        lambda seg: determine_tonic_from_pitches(filter_stable_pitches(seg))[0] if len(filter_stable_pitches(seg)) else None,
        pitches, drone_stable) if drone_stable else None

    return {
        "file": filename,
        "intended_raga": intended_raga,
        "n_pitches": len(pitches),
        "n_stable_pitches": len(stable_pitches),
        "n_hpss_pitches": len(hpss_pitches),
        "n_pyin_pitches": len(pyin_pitches),
        "drone_raw": drone_raw,
        "drone_raw_spread": drone_raw_spread,
        "drone_stable": drone_stable,
        "drone_stable_spread": drone_stable_spread,
        "drone_hpss": drone_hpss,
        "drone_pyin_stable": drone_pyin_stable,
        "drone_pyin": drone_pyin,
        "median": round(median_tonic, 2),
        "targeted_label": round(targeted, 2) if targeted else None,
        "targeted_gap_pct": round(targeted_gap * 100, 1) if targeted_gap is not None else None,
    }


def cents_diff(a, b):
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return round(1200.0 * np.log2(a / b), 0)


def main():
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    classifier = RaagaClassifier(RAGA_REGISTRY)

    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.wav")))
    if not files:
        print(f"No recordings found in {RECORDINGS_DIR}/")
        return

    results = []
    for f in files:
        print(f"Analyzing {f}... (this includes HPSS + continuous pyin, may take a bit)")
        results.append(analyze_one(f, extractor, classifier))

    methods = ["drone_raw", "drone_stable", "drone_hpss", "drone_pyin", "drone_pyin_stable"]

    def fmt(v):
        return f"{v:.2f}" if v else "None"

    print("\n" + "=" * 170)
    print(f"{'File':<32} {'n':>5} " + " ".join(f"{m:>16}" for m in methods) + f" {'targeted*':>10}")
    print("-" * 170)
    for r in results:
        if "drone_raw" not in r:
            print(f"{r['file']:<32} ERROR: {r.get('error')}")
            continue
        print(f"{r['file']:<32} {r['n_pitches']:>5} " +
              " ".join(f"{fmt(r[m]):>16}" for m in methods) + f" {fmt(r['targeted_label']):>10}")
    print("=" * 170)
    print("* targeted_label is DIAGNOSTIC ONLY - uses the filename's raga label, never used live")

    print("\nDistance from label-informed reference, per file (cents, lower = closer):")
    print(f"{'File':<32} " + " ".join(f"{m:>16}" for m in methods))
    for r in results:
        if "drone_raw" not in r or not r.get("targeted_label"):
            continue
        diffs = {m: cents_diff(r[m], r["targeted_label"]) for m in methods}
        print(f"{r['file']:<32} " + " ".join(f"{str(diffs[m]):>16}" for m in methods))

    valid = [r for r in results if r.get("targeted_label")]
    print("\nMean |cents diff| from targeted reference (lower = closer, n = files with a value):")
    for m in methods:
        diffs = [abs(cents_diff(r[m], r["targeted_label"])) for r in valid if r.get(m)]
        if diffs:
            print(f"  {m:<14} {np.mean(diffs):>7.0f}  (n={len(diffs)})")
        else:
            print(f"  {m:<14} no valid values")

    raw_spreads = [r["drone_raw_spread"] for r in results if r.get("drone_raw_spread") is not None]
    stable_spreads = [r["drone_stable_spread"] for r in results if r.get("drone_stable_spread") is not None]
    if raw_spreads or stable_spreads:
        print(f"\nMean within-recording stability spread: drone_raw={np.mean(raw_spreads):.0f} cents (n={len(raw_spreads)}), "
              f"drone_stable={np.mean(stable_spreads):.0f} cents (n={len(stable_spreads)})")


if __name__ == "__main__":
    main()
