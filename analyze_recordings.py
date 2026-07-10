# -*- coding: utf-8 -*-
"""
Re-analyze every saved recording under recordings/ through the new two-phase
live-mode pipeline (see run_live_mode() in main_22.py): the first
LIVE_TONIC_WINDOW_SECONDS worth of audio content decides the tonic via
drone-interval priors (raga-agnostic - doesn't know or use the raga label),
then ALL of the recording's pitches (tonic window included, matching
run_live_mode()'s behavior of folding Phase 1 into the final histogram) are
classified and scored against the raga named in the filename - used purely
as an after-the-fact label for tabulation, never fed into tonic or raga
detection.

Note: save_recording() only keeps non-silent chunks, so a saved file has no
timestamps - "first 2 minutes" here means the first 2 minutes' worth of audio
content by chunk duration, not 2 minutes of original wall-clock session time.

Usage: python3 analyze_recordings.py
"""
import os
import glob
import numpy as np

from main_22 import (
    load_file_chunks, FeatureExtractor, RaagaClassifier, RAGA_REGISTRY,
    determine_tonic_from_pitches, pitch_to_frame_histogram,
    compute_swara_deviation_report,
    SAMPLE_RATE, BUFFER_SIZE, LIVE_TONIC_WINDOW_SECONDS, RECORDINGS_DIR,
)


def compute_stability_spread(pitches, f_tonic, n_segments=4):
    if len(pitches) < n_segments * 10:
        return None
    seg_len = len(pitches) // n_segments
    offsets = []
    for i in range(n_segments):
        seg = pitches[i*seg_len:] if i == n_segments - 1 else pitches[i*seg_len:(i+1)*seg_len]
        seg_tonic = determine_tonic_from_pitches(seg)[0]
        if seg_tonic is not None:
            offsets.append(1200.0 * np.log2(seg_tonic / f_tonic))
    return round(max(offsets) - min(offsets), 0) if offsets else None


def analyze_one(filepath, extractor, classifier):
    filename = os.path.basename(filepath)
    intended_raga = filename.split('_')[0]  # label only - never fed into detection

    chunks = load_file_chunks(filepath)
    n_tonic_chunks = int(LIVE_TONIC_WINDOW_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    tonic_chunks = chunks[:n_tonic_chunks]
    remaining_chunks = chunks[n_tonic_chunks:]

    tonic_pitches = []
    for c in tonic_chunks:
        p = extractor.extract_pitch(c)
        if p and not np.isnan(p):
            tonic_pitches.append(p)

    f_tonic, sa_s, ma_s, pa_s = determine_tonic_from_pitches(tonic_pitches)
    if f_tonic is None:
        return {"file": filename, "intended_raga": intended_raga, "error": "no pitch detected in tonic window"}

    extractor.set_tonic(f_tonic)
    stability_spread = compute_stability_spread(tonic_pitches, f_tonic)

    remaining_pitches = []
    for c in remaining_chunks:
        p = extractor.extract_pitch(c)
        if p and not np.isnan(p):
            remaining_pitches.append(p)

    all_pitches = tonic_pitches + remaining_pitches
    frames = []
    for p in all_pitches:
        fh = pitch_to_frame_histogram(extractor, p)
        if fh is not None:
            frames.append(fh)

    if not frames:
        return {"file": filename, "intended_raga": intended_raga, "tonic_hz": f_tonic,
                "error": "no pitch detected overall"}

    ch = np.mean(frames, axis=0)
    ch /= np.sum(ch) if np.sum(ch) > 0 else 1
    scores = classifier.classify(ch)
    detected_raga = max(scores, key=scores.get)
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    deviation_report = compute_swara_deviation_report(ch, intended_raga)

    return {
        "file": filename,
        "intended_raga": intended_raga,
        "tonic_hz": round(f_tonic, 2),
        "drone_sa_pct": round(sa_s * 100, 1),
        "drone_ma_pct": round(ma_s * 100, 1),
        "drone_pa_pct": round(pa_s * 100, 1),
        "stability_spread_cents": stability_spread,
        "detected_raga": detected_raga,
        "match": detected_raga.lower() == intended_raga.lower(),
        "top3": top3,
        "deviation_report": deviation_report,
        "n_pitches_tonic_window": len(tonic_pitches),
        "n_pitches_total": len(all_pitches),
    }


def main():
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    classifier = RaagaClassifier(RAGA_REGISTRY)

    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.wav")))
    if not files:
        print(f"No recordings found in {RECORDINGS_DIR}/")
        return

    results = []
    for f in files:
        print(f"Analyzing {f}...")
        results.append(analyze_one(f, extractor, classifier))

    print("\n" + "=" * 108)
    print(f"{'File':<34} {'Label':<9} {'Tonic Hz':>9} {'Stab(c)':>8} {'Detected':<10} {'Match':<6} {'Biggest deviation'}")
    print("-" * 108)
    for r in results:
        if "detected_raga" not in r:
            print(f"{r['file']:<34} {r.get('intended_raga', '?'):<9} ERROR: {r.get('error')}")
            continue
        dev = r["deviation_report"][0] if r["deviation_report"] else None
        dev_str = f"{dev['swara']} {dev['deviation_pct']:+.1f}%" if dev else "-"
        match_str = "YES" if r["match"] else "no"
        stab_str = str(r["stability_spread_cents"]) if r["stability_spread_cents"] is not None else "-"
        print(f"{r['file']:<34} {r['intended_raga']:<9} {r['tonic_hz']:>9.2f} {stab_str:>8} "
              f"{r['detected_raga']:<10} {match_str:<6} {dev_str}")
    print("=" * 108)

    scored = [r for r in results if "detected_raga" in r]
    n_matched = sum(1 for r in scored if r["match"])
    print(f"\nOverall: {n_matched}/{len(scored)} matched the labeled raga "
          f"({len(results) - len(scored)} error(s))")

    print("\nPer-file top-3 classifications and biggest deviations:")
    for r in results:
        if "detected_raga" not in r:
            continue
        print(f"\n{r['file']} (label: {r['intended_raga']}, tonic {r['tonic_hz']} Hz, "
              f"drone Sa={r['drone_sa_pct']}% Ma={r['drone_ma_pct']}% Pa={r['drone_pa_pct']}%, "
              f"stability spread {r['stability_spread_cents']} cents)")
        print("  Top 3: " + ", ".join(f"{name}={score*100:.1f}%" for name, score in r["top3"]))
        print("  Biggest deviations:")
        for d in r["deviation_report"][:3]:
            flag = " [FORBIDDEN]" if d["is_forbidden"] and d["actual_pct"] > 2.0 else ""
            print(f"    {d['swara']:<12} actual {d['actual_pct']:>5.1f}% | target {d['target_pct']:>5.1f}% | "
                  f"dev {d['deviation_pct']:+.1f}%{flag}")


if __name__ == "__main__":
    main()
