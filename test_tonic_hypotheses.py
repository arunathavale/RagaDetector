# -*- coding: utf-8 -*-
"""
For each downloaded test file in recordings/ (simple names like "yaman01.wav",
no underscores - distinguishing them from this project's own
{raga}_{artist}_{timestamp}.wav recordings), compares two tonics side by side:

  auto   - continuous pyin + drone-prior search, same method run_file_mode()
           uses automatically today. No singer assumption at all.
  manual - a listening-based judgment call filled in by hand below, after
           actually listening to the file and deciding whether the singer
           sounds male or female (and any other cues). Common Hindustani
           priors: ~140 Hz for male classical vocalists (per several prolific
           singers Arun named - Bhimsen Joshi, Mahesh Kale, Rahul Deshpande -
           in serious classical renditions), ~220 Hz for female - but this is
           just a starting point, not a rule; type whatever you judge fits.

Both columns report a confidence gap (winner's score minus 2nd place, in
points) alongside the detected raga - a small gap means the match is marginal
even when the top pick happens to be correct (this bit us on yaman01.wav: 220
Hz "passed" with only a 23-point gap, while 145 Hz passed with a 51-point gap -
the weaker match would have looked identical to the strong one if only the top
pick were reported).

Classifies against the WHOLE file each time (matching run_file_mode()'s
current behavior - no window exclusion). The raga label is inferred from the
filename (stripped of trailing digits), used only for scoring, never fed into
detection.

Fill in MANUAL_TONIC_HZ below as you go through files by ear. Each entry can
be:
  'male' / 'female'  - shorthand, converted to MALE_TONIC_HZ/FEMALE_TONIC_HZ
                       below (140/220 Hz, averaged from 10 well-known singers
                       each - see PROJECT_PLAN.md's 2026-07-11 entry)
  a number           - an exact Hz value, if you want to fine-tune beyond the
                       gender default
  None               - not listened to yet; skipped in the manual column,
                       not guessed at

Usage: python3 test_tonic_hypotheses.py
"""
import os
import re
import glob
import numpy as np

from main_22 import (
    load_file_chunks, FeatureExtractor, RaagaClassifier, RAGA_REGISTRY, RAGA_SYNONYMS,
    determine_tonic_from_pitches, continuous_pyin_pitches, pitch_to_frame_histogram,
    SAMPLE_RATE, BUFFER_SIZE, LIVE_TONIC_WINDOW_SECONDS, RECORDINGS_DIR,
)

MALE_TONIC_HZ = 140.0    # averaged from 10 well-known male singers (139.69 Hz)
FEMALE_TONIC_HZ = 220.0  # averaged from 10 well-known female singers (216.14 Hz)

MANUAL_TONIC_HZ = {
    'yaman01.wav': None,
    'yaman02.wav': None,
    'asavari01.wav': None,
    'asawari02.wav': None,
    'bhoop01.wav': None,
    'bhoop02.wav': None,
}


def resolve_manual_hz(entry):
    if entry is None:
        return None
    if isinstance(entry, str):
        key = entry.strip().lower()
        if key == 'male':
            return MALE_TONIC_HZ
        if key == 'female':
            return FEMALE_TONIC_HZ
        raise ValueError(f"MANUAL_TONIC_HZ entry {entry!r} must be 'male', 'female', a number, or None")
    return float(entry)


def infer_raga_from_filename(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    name = re.sub(r'\d+$', '', stem).strip().lower()
    if name in RAGA_SYNONYMS:
        return RAGA_SYNONYMS[name]
    match = next((r for r in RAGA_REGISTRY if r.lower() == name), None)
    return match


def is_downloaded_test_file(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return '_' not in stem  # this project's own recordings always have underscores


def classify_with_tonic(chunks, tonic_hz, extractor, classifier):
    extractor.set_tonic(tonic_hz)
    frames = []
    for c in chunks:
        p = extractor.extract_pitch(c)
        if p and not np.isnan(p):
            fh = pitch_to_frame_histogram(extractor, p)
            if fh is not None:
                frames.append(fh)
    if not frames:
        return None, {}, None
    ch = np.mean(frames, axis=0)
    ch /= np.sum(ch) if np.sum(ch) > 0 else 1
    scores = classifier.classify(ch)
    detected = max(scores, key=scores.get)
    others = sorted((s for r, s in scores.items() if r != detected), reverse=True)
    gap_pct = (scores[detected] - others[0]) * 100 if others else 100.0
    return detected, scores, gap_pct


def auto_detect_tonic(chunks, extractor):
    n_tonic_chunks = int(LIVE_TONIC_WINDOW_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    tonic_chunks = chunks[:n_tonic_chunks]
    audio = np.concatenate(tonic_chunks).astype(np.float32) if tonic_chunks else np.array([])
    pitches = continuous_pyin_pitches(audio)
    if not pitches:
        for c in tonic_chunks:
            p = extractor.extract_pitch(c)
            if p and not np.isnan(p):
                pitches.append(p)
    tonic, *_ = determine_tonic_from_pitches(pitches)
    return tonic


def analyze_file(filepath, manual_hz, extractor, classifier):
    chunks = load_file_chunks(filepath)
    result = {}

    auto_tonic = auto_detect_tonic(chunks, extractor)
    if auto_tonic:
        detected, _, gap = classify_with_tonic(chunks, auto_tonic, extractor, classifier)
        result['auto'] = (auto_tonic, detected, gap)
    else:
        result['auto'] = (None, None, None)

    if manual_hz is not None:
        detected, _, gap = classify_with_tonic(chunks, manual_hz, extractor, classifier)
        result['manual'] = (manual_hz, detected, gap)
    else:
        result['manual'] = (None, None, None)

    return result


def main():
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    classifier = RaagaClassifier(RAGA_REGISTRY)

    all_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.wav")))
    files = [f for f in all_files if is_downloaded_test_file(f)]
    if not files:
        print(f"No downloaded test files found in {RECORDINGS_DIR}/ (looking for simple names with no underscore)")
        return

    print("Analyzing files (this includes a pyin pass per file, may take a while)...")
    rows = []
    for f in files:
        raga = infer_raga_from_filename(f)
        if raga is None:
            print(f"{os.path.basename(f):<20} could not infer raga from filename - skipped")
            continue
        manual_hz = resolve_manual_hz(MANUAL_TONIC_HZ.get(os.path.basename(f)))
        print(f"  {f}...{'' if manual_hz is None else f' (manual: {manual_hz} Hz)'}")
        r = analyze_file(f, manual_hz, extractor, classifier)
        rows.append((f, raga, r))

    col = "{:<20} {:<9} {:>8} {:<16} {:>8} {:<16}"
    header = col.format('File', 'Label', 'Auto Hz', 'Auto->', 'Man. Hz', 'Manual->')
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for f, raga, r in rows:
        auto_hz, auto_det, auto_gap = r['auto']
        man_hz, man_det, man_gap = r['manual']

        def fmt(det, gap):
            if det is None:
                return "?"
            marker = "*" if det == raga else ""
            gap_str = f" ({gap:+.0f})" if gap is not None else ""
            return f"{det}{marker}{gap_str}"

        auto_hz_str = f"{auto_hz:.1f}" if auto_hz else "-"
        man_hz_str = f"{man_hz:.1f}" if man_hz else "-"
        print(col.format(os.path.basename(f), raga, auto_hz_str, fmt(auto_det, auto_gap),
                          man_hz_str, fmt(man_det, man_gap)))
    print("=" * len(header))
    print("* = matched the filename-inferred label. (N) = confidence gap in points over 2nd place -")
    print("    a small gap (roughly <30) means the match is marginal even if the top pick is right.")


if __name__ == "__main__":
    main()
