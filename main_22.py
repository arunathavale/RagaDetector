# -*- coding: utf-8 -*-
"""
7-Raga Studio Engine - Real-Time MIR Orchestrator
Features immediate dashboard ignition, correct configuration constants,
and seamless integration with FeatureExtractor for live tracking.
"""

import os
import re
import numpy as np
import time
from collections import deque, Counter
import logging
import json
import sys
import librosa
from scipy.io import wavfile

from audio_stream import AudioStream
from feature_extraction import FeatureExtractor
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    HISTOGRAM_BINS,
    BINS_PER_SEMITONE,
    ROLLING_WINDOW_SECONDS,
    SWARA_MAPPING,
    RAAGA_DATABASE,
    MIN_FREQUENCY,
    MAX_FREQUENCY,
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Standard upper-case swara index variables (same fixed order as config.SWARA_MAPPING)
SA, RE_K, RE, GA_K, GA, MA, MA_T, PA, DHA_K, DHA, NI_K, NI = range(12)
SWARA_NAMES = ['Sa', 'Re_K', 'Re', 'Ga_K', 'Ga', 'Ma', 'Ma_T', 'Pa', 'Dha_K', 'Dha', 'Ni_K', 'Ni']
# Long-form swara names in the same fixed index order, matching config.py's aroha/
# avroha/forbidden_notes naming convention (SWARA_NAMES above is just for display).
LONG_SWARA_NAMES = [name for name, _ in sorted(SWARA_MAPPING.items(), key=lambda kv: kv[1])]

RAGA_REGISTRY = RAAGA_DATABASE

RAGA_SYNONYMS = {"marva": "Marwa", "bhoop": "Bhupali", "bhoopali": "Bhupali", "bhairv": "Bhairav",
                  "asawari": "Asavari"}

TONIC_DETECTION_SECONDS = 5.0
FALLBACK_TONIC_HZ = 145.0
TYPICAL_TONIC_RANGE_HZ = (100.0, 500.0)  # soft plausibility check, not a hard bound
RECORDINGS_DIR = "recordings"
SESSION_DURATION_SECONDS = 240.0  # 4 min - a 120s window risked landing mostly in
# the alap (the slow, exploratory opening) before a performance settles into a
# fuller range of characteristic phrases
LIVE_TONIC_WINDOW_SECONDS = 120.0  # first half of the session: silent tonic
# calculation from the singer's own real performance, instead of a separate blind
# 5-second "sing your Sa" window or a manually-typed guess - both were repeatedly
# wrong this session. A live display can't be trustworthy before a real tonic is
# known, so don't show one until it is.

class RaagaClassifier:
    NOISE_THRESHOLD = 0.05  # per-swara aggregate, not per-bin - see _apply_swara_threshold()

    def __init__(self, db):
        self.db = db

    def _apply_swara_threshold(self, hist):
        """Zero out a swara's whole 10-bin block if its AGGREGATE share is below the
        noise threshold, rather than zeroing individual bins independently. Real
        singing naturally spreads pitch across several bins within a swara's range
        (vibrato, natural pitch wobble) - a per-bin threshold zeroes out legitimate
        signal that's real in aggregate but never concentrated into any single bin,
        which synthetic sine-tone test data never exposed since it doesn't have that
        spread. Found via a real recording where every one of 120 bins sat under 5%
        despite clear per-swara structure (the raga's vadi correctly dominant at
        14%), so the old per-bin threshold zeroed the entire histogram to nothing."""
        cl = hist.copy()
        for i in range(12):
            block = slice(i * BINS_PER_SEMITONE, (i + 1) * BINS_PER_SEMITONE)
            if np.sum(cl[block]) < self.NOISE_THRESHOLD:
                cl[block] = 0.0
        return cl

    def classify(self, hist):
        cl = self._apply_swara_threshold(hist)
        scores = {}
        for name, entry in self.db.items():
            ref = entry["histogram"]
            num, n1, n2 = np.dot(cl, ref), np.linalg.norm(cl), np.linalg.norm(ref)
            sim = (num / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0
            scores[name] = max(0.0, min(1.0, sim))
        mn, mx = min(scores.values()), max(scores.values())
        return {k: ((v - mn) / (mx - mn) if mx > mn else 1.0/len(scores)) for k, v in scores.items()}

def draw_dashboard(elapsed, tonic, scores, intended, live_hist):
    os.system("clear")
    print("=" * 70 + f"\n 🎼 RAGADETECTOR MASTER STUDIO | Track Focus: [{intended}]\n" + "=" * 70)
    print(f" Time Remaining   : {max(0, int(SESSION_DURATION_SECONDS) - int(elapsed))} seconds | Active Tonic (Sa): {tonic} Hz\n" + "-" * 70)

    print(" REAL-TIME 12-SWARA SPECTRUM ANALYZER (Live Mic vs Ground Truth Target):")
    print("-" * 70)
    ref_hist = RAGA_REGISTRY[intended]["histogram"]

    for i, swara in enumerate(SWARA_NAMES):
        live_val = np.sum(live_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])
        ref_val = np.sum(ref_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])

        live_bar = "█" * int(live_val * 20)
        ref_bar  = "░" * int(ref_val * 20)

        print(f"  {swara:<6} | Live: [{live_bar:<20}] ({live_val*100:>4.1f}%) | Target: [{ref_bar:<20}]")

    print("-" * 70)
    print(" Top 3 Nearest Vector Classifications:")
    top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for r, s in top_3:
        print(f"  • {r:<12} [{'█'*int(s*20) + '░'*(20-int(s*20))}] {s*100:>5.1f}%")
    print("=" * 70)

def print_file_summary(duration_s, tonic, scores, intended, live_hist):
    print("=" * 70 + f"\n 🎼 RAGADETECTOR FILE ANALYSIS | Track Focus: [{intended}]\n" + "=" * 70)
    print(f" Audio Duration   : {duration_s:.1f} seconds | Tonic (Sa): {tonic} Hz\n" + "-" * 70)

    print(" 12-SWARA SPECTRUM ANALYZER (File vs Ground Truth Target):")
    print("-" * 70)
    ref_hist = RAGA_REGISTRY[intended]["histogram"]

    for i, swara in enumerate(SWARA_NAMES):
        live_val = np.sum(live_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])
        ref_val = np.sum(ref_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])

        live_bar = "█" * int(live_val * 20)
        ref_bar  = "░" * int(ref_val * 20)

        print(f"  {swara:<6} | File: [{live_bar:<20}] ({live_val*100:>4.1f}%) | Target: [{ref_bar:<20}]")

    print("-" * 70)
    print(" Top 3 Nearest Vector Classifications:")
    top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for r, s in top_3:
        print(f"  • {r:<12} [{'█'*int(s*20) + '░'*(20-int(s*20))}] {s*100:>5.1f}%")
    print("=" * 70)

def compute_swara_deviation_report(actual_hist, intended_raga):
    """Per-swara actual % vs. theoretical target % for the full session, sorted by
    biggest deviation first. This is the same per-swara aggregation draw_dashboard()
    already does every second, just run once over the whole accumulated session
    instead of a single (rolling-window) frame, so it shows where the performance as
    a whole strayed from the target rather than just a live snapshot."""
    entry = RAGA_REGISTRY[intended_raga]
    ref_hist = entry["histogram"]
    forbidden = set(entry.get("forbidden_notes", []))

    report = []
    for i, swara in enumerate(LONG_SWARA_NAMES):
        actual_pct = float(np.sum(actual_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])) * 100
        target_pct = float(np.sum(ref_hist[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE])) * 100
        report.append({
            "swara": swara,
            "actual_pct": round(actual_pct, 1),
            "target_pct": round(target_pct, 1),
            "deviation_pct": round(actual_pct - target_pct, 1),
            "is_forbidden": swara in forbidden,
        })
    report.sort(key=lambda r: abs(r["deviation_pct"]), reverse=True)
    return report

def print_deviation_report(report):
    print("\n" + "-" * 70)
    print(" FULL-SESSION SWARA DEVIATION REPORT (actual vs. theoretical target):")
    print("-" * 70)
    for r in report:
        flag = "  ⚠ FORBIDDEN NOTE" if r["is_forbidden"] and r["actual_pct"] > 2.0 else ""
        sign = "+" if r["deviation_pct"] >= 0 else ""
        print(f"  {r['swara']:<12} actual {r['actual_pct']:>5.1f}% | target {r['target_pct']:>5.1f}% | "
              f"deviation {sign}{r['deviation_pct']:>5.1f}%{flag}")
    print("-" * 70)

def print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s, deviation_report=None, tonic_source=None, artist_name=None):
    meta = RAGA_REGISTRY.get(raga_out, {"vadi": "Unknown", "samvadi": "Unknown", "pakad": []})
    is_match = "PASSED ✅" if raga_out.lower() == intended_raga.lower() else "FAILED ❌"
    if deviation_report:
        print_deviation_report(deviation_report)
    print(f"\n============================================================\n CONCURRENT DETECTED RAAGA : {raga_out}\n VALIDATION LAB STATUS      : {is_match}\n------------------------------------------------------------")
    summary = {
        "session_metadata": {
            "artist": artist_name or "Unknown",
            "execution_duration_seconds": round(duration_s, 2),
            "input_tonic_sa_hz": f_tonic,
            "tonic_source": tonic_source,
        },
        "classification_results": {"detected_dominant_raga": raga_out, "validation_status": is_match},
        "musicological_ground_truth": {
            "raga_name": raga_out,
            "vadi_king_note": meta["vadi"],
            "samvadi_queen_note": meta["samvadi"],
            "pakad_signature_phrase": " -> ".join(meta["pakad"])
        }
    }
    if deviation_report is not None:
        summary["swara_deviation_report"] = deviation_report
    print(json.dumps(summary, indent=2))

def parse_tonic_input(tonic_input):
    """Parse a manually-entered tonic. Returns a float if tonic_input is a valid
    number, or None for anything else (blank, or unparseable text like "auto") -
    both cases mean "auto-detect", not "silently use a default"."""
    if not tonic_input:
        return None
    try:
        return float(tonic_input)
    except ValueError:
        return None

def detect_tonic(extractor, chunks, fallback=FALLBACK_TONIC_HZ):
    """Auto-detect tonic from a batch of audio chunks, falling back to a default on
    failure. Returns (tonic_hz, was_detected) - was_detected=False means the
    fallback was used and the value has NO relationship to the actual audio. This
    matters because FALLBACK_TONIC_HZ (145.0) happens to sit inside
    TYPICAL_TONIC_RANGE_HZ, so a failed detection looks just as "plausible" as a
    real one to the soft range check in confirm_or_override_tonic() - without
    tracking success/failure explicitly, a silent fallback is indistinguishable
    from a genuine detection by the time it reaches the user."""
    if not chunks:
        print(f"No audio captured for tonic detection; using fallback {fallback} Hz")
        return fallback, False
    tonic = extractor.auto_detect_tonic(chunks, method='median')
    if tonic is None:
        print(f"Could not detect a tonic from the audio; using fallback {fallback} Hz")
        return fallback, False
    print(f"Auto-detected tonic: {tonic} Hz")
    return tonic, True

def confirm_or_override_tonic(f_tonic, was_detected=True, retry_fn=None):
    """Auto-detection has no way to know whether the calibration window actually
    captured the intended Sa (e.g. an instrumental intro playing instead of a held
    vocal note produces a plausible-looking but meaningless value) - surface the
    result and let the user catch an obviously wrong one before it silently corrupts
    the whole session, rather than barreling ahead on blind trust.

    Returns (tonic_hz, source) where source is 'auto_detected', 'fallback_accepted',
    or 'manual_override' - callers should record this, not just the number, since a
    bare Hz value alone can't distinguish "genuinely detected" from "accepted a
    meaningless fallback that happened to look plausible"."""
    retry_hint = ", or 'r' to re-listen" if retry_fn else ""
    while True:
        if not was_detected:
            print(f"\n🚫 No pitch could be detected at all - {f_tonic:.2f} Hz is just the "
                  f"hardcoded fallback default, with NO relationship to your actual voice "
                  f"or performance. Strongly recommend typing a real number{' or retrying' if retry_fn else ''} "
                  f"rather than accepting this.")
        else:
            lo, hi = TYPICAL_TONIC_RANGE_HZ
            if not (lo <= f_tonic <= hi):
                print(f"\n⚠️  {f_tonic:.2f} Hz is outside the typical vocal tonic range "
                      f"({lo:.0f}-{hi:.0f} Hz) - this often means the calibration window "
                      f"caught something other than a held Sa (an instrumental intro, "
                      f"wrong timing, etc.), not your actual tonic.")
        answer = input(f"Using tonic {f_tonic:.2f} Hz - press Enter to accept, "
                        f"type a number to override{retry_hint}: ").strip()
        if not answer:
            return f_tonic, ('auto_detected' if was_detected else 'fallback_accepted')
        if retry_fn and answer.lower() in ('r', 'retry'):
            f_tonic, was_detected = retry_fn()
            continue
        try:
            return float(answer), 'manual_override'
        except ValueError:
            print(f"'{answer}' isn't a number{' or r' if retry_fn else ''} - try again.")

def load_file_chunks(filepath, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE):
    """Load an audio file and split it into BUFFER_SIZE-sized chunks, resampled to match the mic pipeline."""
    audio, _ = librosa.load(filepath, sr=sample_rate, mono=True)
    return [audio[i:i + buffer_size] for i in range(0, len(audio), buffer_size)]

def save_recording(chunks, intended_raga, artist_name=None, sample_rate=SAMPLE_RATE):
    """Persist the raw session audio to disk so a live session can be re-analyzed
    later (different pitch-detection methods, building empirical reference
    histograms, etc.) without needing to re-run it - live audio is otherwise
    processed and discarded in real time with no record kept. Note: this only
    contains the non-silent chunks AudioStream's adaptive threshold already let
    through, not a literal unedited recording including silence gaps.

    Filename encodes the artist (sanitized, defaulting to "Unknown") alongside
    the raga - added after a session spent significant time re-listening to old
    recordings to figure out after the fact who was actually singing (several
    turned out to be Bollywood songs or interrupted broadcasts, not usable
    classical performances at all). Capturing this up front avoids repeating
    that forensic work on every future batch of recordings."""
    if not chunks:
        return None
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    audio = np.concatenate(chunks).astype(np.float32)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_artist = re.sub(r'[^A-Za-z0-9]+', '', artist_name) if artist_name else "Unknown"
    filepath = os.path.join(RECORDINGS_DIR, f"{intended_raga}_{safe_artist}_{timestamp}.wav")
    wavfile.write(filepath, sample_rate, audio)
    print(f"\nSaved session recording to {filepath}")
    return filepath

def subtract_swara_noise_floor(hist_120, bins_per_semitone=BINS_PER_SEMITONE):
    """Estimate a per-session noise floor as the smallest of the 12 swara-aggregated
    shares, subtract that floor uniformly from every swara's share, and renormalize
    back to 1.0. Operates on the 120-bin histogram by scaling each swara's block of
    bins down proportionally, so the result stays a valid 120-bin histogram the
    existing classifier can use directly.

    Caveat: this assumes contamination is roughly uniform across all 12 swara. It
    probably isn't entirely - tanpura specifically reinforces Sa (and often Pa), so
    those two likely carry a bigger, non-uniform bump than a flat noise floor would
    predict. Expect this to help some, not fully solve the accompaniment problem."""
    swara_totals = np.array([
        np.sum(hist_120[i*bins_per_semitone:(i+1)*bins_per_semitone])
        for i in range(12)
    ])
    floor = np.min(swara_totals)
    adjusted_totals = np.clip(swara_totals - floor, 0, None)
    result = np.zeros_like(hist_120)
    for i in range(12):
        old_total = swara_totals[i]
        if old_total > 0:
            block = hist_120[i*bins_per_semitone:(i+1)*bins_per_semitone]
            result[i*bins_per_semitone:(i+1)*bins_per_semitone] = block * (adjusted_totals[i] / old_total)
    total = np.sum(result)
    return result / total if total > 0 else result

def pitch_to_frame_histogram(extractor, pitch):
    """One-hot 120-bin histogram for a single detected pitch. Deliberately does not
    use extractor.add_to_histogram()/get_normalized_histogram() - those accumulate
    into a single running total for the extractor's entire lifetime with no reset or
    decay, so a "current" reading taken from them drifts less and less over a long
    session (early singing ends up permanently baked in). Building one frame per
    detection here and averaging over a bounded deque is what actually gives a live
    rolling window."""
    cents = extractor.frequency_to_cents(pitch)
    wrapped = extractor.wrap_to_octave(cents)
    bin_index = extractor.cents_to_bin(wrapped)
    if bin_index is None:
        return None
    frame = np.zeros(HISTOGRAM_BINS)
    frame[bin_index] = 1.0
    return frame

MIN_NYAS_FRAMES = 15  # see extract_nyas_sequence() - roughly 0.3-0.4s of BUFFER_SIZE
# chunks at SAMPLE_RATE, a rough floor for "genuinely held" vs. a passing glide tone
NYAS_SMOOTHING_WINDOW = 5  # see _smooth_swara_sequence()

def _smooth_swara_sequence(swara_indices, window_size=NYAS_SMOOTHING_WINDOW):
    """Median/mode filter over the raw per-frame swara sequence: replace each
    frame's value with the most common value in a window centered on it.
    Standard technique for removing brief single-frame noise spikes while
    preserving genuine, sustained transitions.

    Needed because extract_nyas_sequence()'s run-length grouping was found to
    be far too brittle on real audio: it requires a run of PERFECTLY
    consecutive identical-swara frames, with zero tolerance for even one
    stray frame. Clean synthetic sine tones never produce a stray frame mid-
    note, so this worked fine there (validated with exact pakad reconstruction
    - see PROJECT_PLAN.md's 2026-07-11 entry) - but real singing has natural
    vibrato/micro-wobble that flickers a frame into a neighboring bin even
    while a singer is clearly, genuinely holding a note, which reset the run
    count to zero every time. Confirmed directly: one real 4.5-minute
    recording with over 10,000 raw pitch detections produced only 12 nyas
    events before this fix, and another produced just 1 - the exact "works on
    synthetic, breaks on real audio" pattern that has bitten this project
    multiple times this session (the per-bin histogram bug, the pitch-
    detector octave-error bug)."""
    n = len(swara_indices)
    if n == 0:
        return []
    half = window_size // 2
    smoothed = []
    for i in range(n):
        window = [s for s in swara_indices[max(0, i - half):min(n, i + half + 1)] if s is not None]
        smoothed.append(Counter(window).most_common(1)[0][0] if window else None)
    return smoothed

def extract_nyas_sequence(extractor, pitches, min_nyas_frames=MIN_NYAS_FRAMES):
    """Convert a time-ordered pitch stream into a sequence of discrete swara
    (nyas - held/resting note) events, instead of the flat, order-blind
    histogram every other classifier in this project uses. Smooths the raw
    per-frame swara sequence (see _smooth_swara_sequence()) to absorb natural
    vibrato/detection noise, then groups consecutive same-swara detections
    into runs and keeps only runs at least min_nyas_frames long - a real
    vocal glide/meend passes through many swara bins in quick succession
    (each individual bin's run is short), while a genuinely held note
    produces a long run of consecutive same-swara frames. This is the
    foundation aroha/avaroha/pakad sequence matching needs (you can't match a
    melodic phrase against a flat bag of notes) and, as a side effect,
    filters out exactly the kind of ornamentation-driven false notes
    diagnosed this session (e.g. Kaushiki Chakraborty's Yaman recording
    reading heavy contamination across every komal note simultaneously,
    consistent with meend transiting through them rather than resting there).

    Requires the extractor's tonic to already be set - swara identity is
    tonic-relative. Repeated returns to the same swara (e.g. Yaman's pakad
    visiting Sa twice) are intentionally kept as separate sequence entries,
    not merged - aroha/avaroha/pakad data itself is written with meaningful
    repeated notes, not a deduplicated set.

    Returns a list of (swara_index, frame_count) tuples, swara_index in
    config.SWARA_MAPPING's fixed 0-11 order (frame_count is a rough duration
    proxy - a count of consecutive detections, not converted to seconds,
    since chunk timing isn't perfectly uniform across silence-gated capture)."""
    if not pitches:
        return []

    raw_swara_indices = []
    for p in pitches:
        cents = extractor.frequency_to_cents(p)
        wrapped = extractor.wrap_to_octave(cents)
        bin_index = extractor.cents_to_bin(wrapped)
        raw_swara_indices.append(bin_index // BINS_PER_SEMITONE if bin_index is not None else None)
    swara_indices = _smooth_swara_sequence(raw_swara_indices)

    sequence = []
    run_swara, run_length = None, 0
    for s in swara_indices:
        if s == run_swara:
            run_length += 1
        else:
            if run_swara is not None and run_length >= min_nyas_frames:
                sequence.append((run_swara, run_length))
            run_swara, run_length = s, 1
    if run_swara is not None and run_length >= min_nyas_frames:
        sequence.append((run_swara, run_length))

    return sequence

def _raga_sequence_to_indices(entry, key):
    """config.RAAGA_DATABASE's aroha/avroha/pakad lists are swara NAMES
    (strings); convert to the same 0-11 index space extract_nyas_sequence()
    uses, dropping anything not in SWARA_MAPPING (defensive - all entries are
    expected to be valid names)."""
    return [SWARA_MAPPING[s] for s in entry.get(key, []) if s in SWARA_MAPPING]

def build_raga_transitions(entry):
    """The set of (from_swara_index, to_swara_index) bigrams that occur
    anywhere in a raga's aroha, avroha, or pakad - i.e. which note-to-note
    moves are actually part of this raga's melodic vocabulary. Used to score
    how consistent an OBSERVED sequence's own transitions are with a
    candidate raga, the way a flat histogram never can (it has no concept of
    "this note followed that one").

    A row-normalized transition PROBABILITY matrix (matching Bhattacharjee &
    Sriniwasan 2011's technique - see PROJECT_PLAN.md's 2026-07-11 literature-
    check entry) was tried in place of this binary set, on the theory that a
    raga's DOMINANT move from a note should count for more than a move it
    merely can make occasionally - exactly the nuance needed for Asavari/
    Jaunpuri's aroha, which share 4 of 5 transitions. Measured WORSE on
    eval_sequence_harness.py (72.0% vs. this binary version's 83.1%), and
    Laplace smoothing to address suspected sparse-data overconfidence made it
    worse still (68.8%) - reverted rather than kept on the strength of the
    literature reference alone without evidence it actually helps here."""
    transitions = set()
    for key in ('aroha', 'avroha', 'pakad'):
        indices = _raga_sequence_to_indices(entry, key)
        for i in range(len(indices) - 1):
            transitions.add((indices[i], indices[i + 1]))
    return transitions

def subsequence_match_ratio(observed_indices, target_indices, max_span_multiplier=3.0, min_match_fraction=0.5):
    """Classic longest-common-subsequence-style check: how much of
    target_indices appears IN ORDER (not necessarily contiguous - other notes
    can appear in between) within observed_indices. Returns the fraction of
    target_indices successfully matched, in [0, 1], scaled down by a SPAN
    PENALTY if the matched notes end up spread far wider across
    observed_indices than the pakad's own length would justify, and zeroed
    out entirely if fewer than min_match_fraction of the pakad's notes were
    found at all.

    Both corrections were necessary, not optional - found on real data (see
    PROJECT_PLAN.md's 2026-07-11 entry) that a naive version has TWO distinct
    ways a short pakad (e.g. Puriya's, 4 notes) unfairly outscores a long one
    (e.g. Bhairav's, 11 notes) purely by chance, not genuine resemblance:
    (1) a FULL match spread across most of a long observed sequence - fixed
    by the span penalty; (2) a PARTIAL match of just one or two notes (a
    single common note like Sa appearing somewhere is nowhere near "the
    pakad was sung," but the plain ratio still gave it credit, with an
    almost-zero span since so few notes were found to spread out at all,
    so the span penalty alone didn't catch it) - fixed by the minimum-match-
    fraction floor. Both problems were the actual cause of a wrong raga
    outscoring the correct one on two different real recordings this
    session, not merely a synthetic-benchmark curiosity."""
    if not target_indices:
        return 0.0
    j = 0
    first_match_pos, last_match_pos = None, None
    for pos, note in enumerate(observed_indices):
        if j < len(target_indices) and note == target_indices[j]:
            if first_match_pos is None:
                first_match_pos = pos
            last_match_pos = pos
            j += 1
    if j == 0:
        return 0.0
    match_ratio = j / len(target_indices)
    if match_ratio < min_match_fraction:
        return 0.0
    span = last_match_pos - first_match_pos + 1
    max_allowed_span = len(target_indices) * max_span_multiplier
    if span > max_allowed_span:
        match_ratio *= max_allowed_span / span
    return match_ratio

class SequenceClassifier:
    """Alternative to RaagaClassifier that scores a nyas (held-note) SEQUENCE
    against each raga's aroha/avroha/pakad, instead of a flat, order-blind
    histogram. Combines two signals per raga: (1) a duration-weighted fraction
    of the observed sequence's own note-to-note transitions that are in this
    raga's melodic vocabulary (see build_raga_transitions()), and (2) how much
    of the raga's specific pakad phrase shows up as an in-order subsequence of
    what was sung. Built to address real failures the histogram approach hit
    in practice this session - sibling ragas sharing a note set (Asavari/
    Jaunpuri, Marwa/Puriya) and even a same-thaat-adjacent pair (Bhupali/
    Yaman) tying at the best possible tonic - all cases where note ORDER is
    the only thing that actually distinguishes the ragas.

    Duration weighting (each transition's contribution is proportional to how
    long its two notes were held, in nyas frame_count) was motivated by a
    finding in Koduri/Gulati/Rao's 2011 survey (see PROJECT_PLAN.md):
    weighting swara occurrence by total held duration, not just instance
    count, gave their best accuracy - a longer-held note is a more confident
    signal than one that barely cleared MIN_NYAS_FRAMES. Measured neutral on
    eval_sequence_harness.py (83.1% either way - that synthetic benchmark
    uses fairly uniform note durations, so there's little real variation for
    this to exploit), kept anyway since it's free on this data and plausibly
    matters more on real audio where singers genuinely linger unevenly."""

    TRANSITION_WEIGHT = 0.5
    PAKAD_WEIGHT = 0.5

    def __init__(self, db):
        self.db = db
        self._transitions_cache = {name: build_raga_transitions(entry) for name, entry in db.items()}

    def classify(self, nyas_sequence):
        """nyas_sequence: output of extract_nyas_sequence() - a list of
        (swara_index, frame_count) tuples. Returns {raga_name: score in [0,1]}
        for every raga in the database (NOT min-max normalized like
        RaagaClassifier.classify() - these are absolute, comparable-across-
        calls scores, so a weak match against every raga looks weak in every
        raga's score, rather than one of them being inflated to 100%)."""
        scores = {}
        for name, entry in self.db.items():
            if len(nyas_sequence) < 2:
                scores[name] = 0.0
                continue
            transitions = self._transitions_cache[name]
            weighted_hit_sum, total_weight = 0.0, 0.0
            for i in range(len(nyas_sequence) - 1):
                from_swara, from_duration = nyas_sequence[i]
                to_swara, to_duration = nyas_sequence[i + 1]
                weight = from_duration + to_duration
                weighted_hit_sum += weight * (1.0 if (from_swara, to_swara) in transitions else 0.0)
                total_weight += weight
            transition_score = weighted_hit_sum / total_weight if total_weight > 0 else 0.0

            observed_indices = [s for s, _ in nyas_sequence]
            pakad_indices = _raga_sequence_to_indices(entry, 'pakad')
            pakad_score = subsequence_match_ratio(observed_indices, pakad_indices)

            scores[name] = self.TRANSITION_WEIGHT * transition_score + self.PAKAD_WEIGHT * pakad_score
        return scores

DRONE_FOURTH_SEMITONE = 5   # Ma_shuddha - Sa-Ma drone dyad
DRONE_FIFTH_SEMITONE = 7    # Pa - Sa-Pa drone dyad, more common Hindustani tuning
DRONE_FOURTH_WEIGHT = 0.5
DRONE_FIFTH_WEIGHT = 0.8
STABILITY_FILTER_CENTS = 25.0  # see filter_stable_pitches()

def filter_stable_pitches(pitches, stability_threshold_cents=STABILITY_FILTER_CENTS):
    """Keep only pitches that sit within stability_threshold_cents of BOTH
    immediate neighbors in the detection sequence - i.e. part of a sustained,
    non-gliding run, not a single noisy blip or a note in the middle of a
    transition. A tanpura drone is steady by definition; a sung phrase glides
    continuously between notes. Raw pitch mass (what determine_tonic_from_pitches()
    uses by default) counts a drone frame and a mid-glide frame identically, so
    it's really measuring wherever the voice spends the most raw time, not
    necessarily the drone - this is a sharper filter for "this frame plausibly
    reflects a held tone" than mass alone.

    Sequence order matters here and is assumed to already be capture order (the
    order pitches were appended to the list as chunks arrived), not sorted by
    value - the same assumption print_tonic_stability_check() makes about its
    input."""
    if len(pitches) < 3:
        return list(pitches)
    stable = []
    for i in range(1, len(pitches) - 1):
        prev_cents = abs(1200.0 * np.log2(pitches[i] / pitches[i - 1]))
        next_cents = abs(1200.0 * np.log2(pitches[i + 1] / pitches[i]))
        if prev_cents <= stability_threshold_cents and next_cents <= stability_threshold_cents:
            stable.append(pitches[i])
    return stable

def determine_tonic_from_pitches(pitches, search_min_hz=TYPICAL_TONIC_RANGE_HZ[0],
                                  search_max_hz=TYPICAL_TONIC_RANGE_HZ[1]):
    """Find the tonic using drone-interval priors: a tanpura almost always drones
    on Sa+Pa (fifth, 700 cents) or Sa+Ma (fourth, 500 cents) - a structural signal
    from how the accompanying instrument is TUNED, not from what's being sung, so
    it works without knowing the raga in advance.

    Deliberately does NOT take the intended/target raga as input, even though one
    is asked for at session start and is available here. An earlier version of
    this function searched for "the tonic that makes raga X classify best" - but
    the ultimate goal is raga IDENTIFICATION, where the raga is exactly the
    unknown being detected. That version only ever validated correctly because it
    was handed the true answer as a label; using it here would be circular and
    silently rely on information the real use case doesn't have. The
    intended-raga input is legitimately used elsewhere (recording filename,
    after-the-fact PASS/FAIL scoring, theoretical-histogram comparison in the
    dashboard) - just never fed into detection itself, tonic or raga.

    Exploits the same circular-shift property as the file-search tools in this
    project (changing tonic is a rotation of the pitch-class histogram: cents =
    1200*log2(f/tonic) mod 1200). Same technique as find_optimal_tonic.py's
    score_drone_priors()/find_best_tonic_by_drone_priors(), reimplemented
    directly here (rather than imported) since find_optimal_tonic.py imports
    from this module and importing back would be circular.

    Returns (tonic_hz, sa_strength, ma_strength, pa_strength), or all-None if no
    pitches were provided."""
    if not pitches:
        return None, None, None, None

    cents_per_bin = 1200.0 / HISTOGRAM_BINS
    anchor_hz = 1.0
    anchor_hist = np.zeros(HISTOGRAM_BINS)
    for p in pitches:
        cents = (1200.0 * np.log2(p / anchor_hz)) % 1200.0
        b = int(cents / cents_per_bin) % HISTOGRAM_BINS
        anchor_hist[b] += 1
    total = np.sum(anchor_hist)
    if total == 0:
        return None, None, None, None
    anchor_hist /= total

    lowest_octave_hz = anchor_hz * (2 ** np.floor(np.log2(search_min_hz / anchor_hz)))
    best_tonic, best_score, best_components = None, -1.0, (0.0, 0.0, 0.0)
    hz = lowest_octave_hz
    while hz <= search_max_hz * 2:
        for shift in range(HISTOGRAM_BINS):
            tonic_hz = hz * (2 ** (shift * cents_per_bin / 1200.0))
            if search_min_hz <= tonic_hz <= search_max_hz:
                hist = np.roll(anchor_hist, -shift)
                sa = np.sum(hist[0:BINS_PER_SEMITONE])
                ma = np.sum(hist[DRONE_FOURTH_SEMITONE*BINS_PER_SEMITONE:(DRONE_FOURTH_SEMITONE+1)*BINS_PER_SEMITONE])
                pa = np.sum(hist[DRONE_FIFTH_SEMITONE*BINS_PER_SEMITONE:(DRONE_FIFTH_SEMITONE+1)*BINS_PER_SEMITONE])
                score = sa + max(DRONE_FOURTH_WEIGHT * ma, DRONE_FIFTH_WEIGHT * pa)
                if score > best_score:
                    best_score, best_tonic, best_components = score, tonic_hz, (sa, ma, pa)
        hz *= 2
    return (best_tonic,) + best_components

PYIN_HOP_LENGTH = 441  # ~10ms at 44100Hz

def continuous_pyin_pitches(raw_audio, sample_rate=SAMPLE_RATE, hop_length=PYIN_HOP_LENGTH,
                             voiced_prob_threshold=0.5):
    """Run librosa.pyin() once, continuously, across a whole audio buffer with a
    proper ~10ms hop - not per-BUFFER_SIZE-chunk like the default autocorrelation
    detector, since pyin's Viterbi decoding needs real temporal context to work as
    designed. Used ONLY for the Phase 1 tonic calculation in run_live_mode(), not
    for live swara tracking - pyin was ruled out there earlier this session for
    being source-agnostic (confident about any stable pitch, not specifically the
    voice), but that same property turns out to be an asset for finding a tanpura
    drone specifically. Compared against plain per-chunk autocorrelation and
    against drone-prior search restricted to locally-stable frames only
    (filter_stable_pitches()) on all 10 saved recordings: continuous pyin gave the
    closest tonic to a classifier-validated reference on average (~35% closer than
    the raw per-chunk method), clearly ahead of every other method tried. See
    compare_tonic_methods.py and PROJECT_PLAN.md's 2026-07-10 entries for the full
    comparison - still far from solved (average error was still ~3 semitones
    even with this best-available method), but a clear, measured improvement."""
    if len(raw_audio) == 0:
        return []
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=raw_audio, fmin=MIN_FREQUENCY, fmax=MAX_FREQUENCY, sr=sample_rate, hop_length=hop_length,
    )
    valid = voiced_flag & (voiced_probs >= voiced_prob_threshold) & np.isfinite(f0)
    return list(f0[valid])

def print_tonic_stability_check(pitches, f_tonic, n_segments=4):
    """Split Phase 1's pitches into n_segments equal, time-ordered chunks (the list
    is already in capture order) and independently drone-tonic each one. Tonic
    essentially never changes mid-performance, so disagreement across segments
    flags a pitch-tracking problem in that portion, not a real tonic shift - same
    diagnostic as find_optimal_tonic.py's check_tonic_stability(), reimplemented
    locally here for the same reason as determine_tonic_from_pitches() above."""
    if len(pitches) < n_segments * 10:
        return
    seg_len = len(pitches) // n_segments
    print("\n Tonic stability check (segment-by-segment, drone priors):")
    cents_offsets = []
    for i in range(n_segments):
        seg = pitches[i*seg_len:] if i == n_segments - 1 else pitches[i*seg_len:(i+1)*seg_len]
        seg_tonic = determine_tonic_from_pitches(seg)[0]
        if seg_tonic is not None:
            cents_off = 1200.0 * np.log2(seg_tonic / f_tonic)
            cents_offsets.append(cents_off)
            print(f"   Segment {i+1}: {seg_tonic:.2f} Hz ({cents_off:+.0f} cents vs. overall)")
    if cents_offsets:
        spread = max(cents_offsets) - min(cents_offsets)
        flag = " ⚠️  meaningful disagreement" if spread > 50 else " (consistent)"
        print(f"   Max spread: {spread:.0f} cents{flag}")

def run_live_mode(extractor, classifier, intended_raga, artist_name=None):
    streamer = AudioStream(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE)
    streamer.start()

    recorded_chunks = []  # every chunk for the whole session (tonic-calculation
    # phase + live-display phase), saved to disk at the end regardless of pass/fail

    # Phase 1: LIVE_TONIC_WINDOW_SECONDS of silent listening. No manual entry, no
    # separate blind "sing your Sa" window - both were repeatedly wrong earlier this
    # session. Instead, accumulate real pitches from the singer's actual performance
    # and calculate the tonic that best fits the already-known intended raga.
    print(f"\nListening for {int(LIVE_TONIC_WINDOW_SECONDS)}s to calculate your tonic from "
          f"actual singing (sing normally - no live display until this finishes)...")
    raw_pitches = []       # per-chunk autocorrelation - used for the swara/deviation
    # content of Phase 1 (folded into history/full_session_frames below), NOT for
    # tonic determination itself - see phase1_chunks/continuous_pyin_pitches() below.
    phase1_chunks = []     # raw audio from Phase 1, for the pyin-based tonic search
    phase1_start = time.time()
    while time.time() - phase1_start < LIVE_TONIC_WINDOW_SECONDS:
        remaining = max(0, int(LIVE_TONIC_WINDOW_SECONDS - (time.time() - phase1_start)))
        print(f"  Calculating tonic... {remaining}s remaining, {len(raw_pitches)} pitch samples so far   ", end='\r')
        while len(streamer.audio_queue) > 0:
            c = streamer.audio_queue.popleft()
            recorded_chunks.append(c)
            phase1_chunks.append(c)
            p = extractor.extract_pitch(c)
            if p and not np.isnan(p):
                raw_pitches.append(p)
        time.sleep(0.1)

    print(f"\nAnalyzing tonic (running pyin over the full {int(LIVE_TONIC_WINDOW_SECONDS)}s window)...")
    phase1_audio = np.concatenate(phase1_chunks).astype(np.float32) if phase1_chunks else np.array([])
    tonic_pitches = continuous_pyin_pitches(phase1_audio)
    if not tonic_pitches:
        tonic_pitches = raw_pitches  # pyin found nothing at all - fall back to the
        # per-chunk autocorrelation pitches rather than declaring total failure

    f_tonic, sa_s, ma_s, pa_s = determine_tonic_from_pitches(tonic_pitches)
    if f_tonic is None:
        f_tonic, tonic_source = FALLBACK_TONIC_HZ, 'fallback_accepted'
        print(f"\n🚫 No pitch could be detected in the first {int(LIVE_TONIC_WINDOW_SECONDS)}s - "
              f"using fallback {FALLBACK_TONIC_HZ} Hz, which has NO relationship to your actual voice.")
    else:
        tonic_source = 'auto_from_recording'
        lo, hi = TYPICAL_TONIC_RANGE_HZ
        print(f"\nCalculated tonic: {f_tonic:.2f} Hz (drone strength: Sa={sa_s*100:.1f}% "
              f"Ma={ma_s*100:.1f}% Pa={pa_s*100:.1f}%)")
        if not (lo <= f_tonic <= hi):
            print(f"⚠️  {f_tonic:.2f} Hz is outside the typical vocal tonic range ({lo:.0f}-{hi:.0f} Hz) - "
                  f"worth a second look if this session's results seem off.")
        print_tonic_stability_check(tonic_pitches, f_tonic)

    extractor.set_tonic(f_tonic)

    history = deque(maxlen=int(ROLLING_WINDOW_SECONDS * (SAMPLE_RATE / BUFFER_SIZE)))
    full_session_frames = []  # unwindowed - every detection for the whole session,
    # for the end-of-run deviation report (history above is deliberately bounded for
    # live responsiveness and isn't representative of the full performance)

    # Phase 1's audio is genuine performance content, not throwaway calibration -
    # fold it into the session's classification data now that tonic is known,
    # rather than discarding it the way the old calibration window did.
    for p in raw_pitches:
        frame_hist = pitch_to_frame_histogram(extractor, p)
        if frame_hist is not None:
            history.append(frame_hist)
            full_session_frames.append(frame_hist)

    raga_out = "Undetermined"
    s_out = {}

    empty_hist = np.zeros(HISTOGRAM_BINS)
    empty_scores = {k: 0.0 for k in RAGA_REGISTRY.keys()}
    draw_dashboard(LIVE_TONIC_WINDOW_SECONDS, f_tonic, empty_scores, intended_raga, empty_hist)

    try:
        while True:
            dt = time.time() - phase1_start
            if dt >= SESSION_DURATION_SECONDS: break

            queue_size = len(streamer.audio_queue)
            if queue_size > 0:
                print(f"[DEBUG] Queue size: {queue_size}", end='\r')

            while len(streamer.audio_queue) > 0:
                c = streamer.audio_queue.popleft()
                recorded_chunks.append(c)
                p = extractor.extract_pitch(c)
                if p and not np.isnan(p):
                    frame_hist = pitch_to_frame_histogram(extractor, p)
                    if frame_hist is not None:
                        history.append(frame_hist)
                        full_session_frames.append(frame_hist)

            if len(history) > 0:
                ch = np.mean(history, axis=0)
                ch /= np.sum(ch) if np.sum(ch) > 0 else 1
                s_out = classifier.classify(ch)
                if s_out and max(s_out.values()) > 0.0:
                    raga_out = max(s_out, key=s_out.get)
                draw_dashboard(dt, f_tonic, s_out, intended_raga, ch)
            else:
                draw_dashboard(dt, f_tonic, empty_scores, intended_raga, empty_hist)

            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.is_running = False
        if hasattr(streamer, "stream") and streamer.stream:
            try:
                streamer.stream.stop_stream()
                streamer.stream.close()
            except:
                pass
        time.sleep(0.2)
        os.system("clear")
        deviation_report = None
        if full_session_frames:
            full_ch = np.mean(full_session_frames, axis=0)
            full_ch /= np.sum(full_ch) if np.sum(full_ch) > 0 else 1
            deviation_report = compute_swara_deviation_report(full_ch, intended_raga)
        save_recording(recorded_chunks, intended_raga, artist_name=artist_name)
        print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s=SESSION_DURATION_SECONDS,
                             deviation_report=deviation_report, tonic_source=tonic_source, artist_name=artist_name)

def run_file_mode(file_path, tonic_input, extractor, classifier, intended_raga, artist_name=None):
    print(f"\nLoading audio file: {file_path}")
    chunks = load_file_chunks(file_path)
    if not chunks:
        print("\n❌ ERROR: could not read any audio from the file.\n")
        sys.exit(1)

    # File mode has the whole file available upfront, so tonic detection uses the
    # same continuous-pyin approach validated for live mode's Phase 1 (see
    # continuous_pyin_pitches()'s docstring - ~35% closer to a validated reference
    # than plain per-chunk autocorrelation), over up to LIVE_TONIC_WINDOW_SECONDS of
    # audio, rather than the old 5-second median-based auto-detect. That old
    # approach assumed the file's opening seconds were a deliberately-held "Sa"
    # calibration reference and excluded them from the classification histogram -
    # true for our own now-retired live-recording ritual, but not for arbitrary
    # downloaded files (a raga performance starting immediately, no calibration
    # tone at all) or current-style live recordings (which no longer have that
    # ritual either). So the opening window is no longer excluded from the
    # histogram - it's genuine performance content, same reasoning as live mode.
    n_tonic_chunks = int(LIVE_TONIC_WINDOW_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
    tonic_chunks = chunks[:n_tonic_chunks]

    manual_tonic = parse_tonic_input(tonic_input)
    if manual_tonic is not None:
        f_tonic, tonic_source = manual_tonic, 'manual'
    else:
        if tonic_input:
            print(f"\n'{tonic_input}' isn't a number - auto-detecting tonic instead.")
        print(f"Analyzing tonic (running pyin over up to {int(LIVE_TONIC_WINDOW_SECONDS)}s of audio)...")
        tonic_window_audio = np.concatenate(tonic_chunks).astype(np.float32) if tonic_chunks else np.array([])
        tonic_pitches = continuous_pyin_pitches(tonic_window_audio)
        if not tonic_pitches:
            for c in tonic_chunks:
                p = extractor.extract_pitch(c)
                if p and not np.isnan(p):
                    tonic_pitches.append(p)

        detected_tonic, sa_s, ma_s, pa_s = determine_tonic_from_pitches(tonic_pitches)
        was_detected = detected_tonic is not None
        if was_detected:
            print(f"Auto-detected tonic: {detected_tonic:.2f} Hz (drone strength: Sa={sa_s*100:.1f}% "
                  f"Ma={ma_s*100:.1f}% Pa={pa_s*100:.1f}%)")
            print_tonic_stability_check(tonic_pitches, detected_tonic)
        else:
            detected_tonic = FALLBACK_TONIC_HZ
        f_tonic, tonic_source = confirm_or_override_tonic(detected_tonic, was_detected)  # no retry:
        # re-reading the same opening slice of the file would just return the identical value

    extractor.set_tonic(f_tonic)
    history = []
    for chunk in chunks:
        p = extractor.extract_pitch(chunk)
        if p and not np.isnan(p):
            frame_hist = pitch_to_frame_histogram(extractor, p)
            if frame_hist is not None:
                history.append(frame_hist)

    duration_s = len(chunks) * BUFFER_SIZE / SAMPLE_RATE
    raga_out = "Undetermined"
    s_out = {}
    deviation_report = None
    if history:
        ch = np.mean(history, axis=0)
        ch /= np.sum(ch) if np.sum(ch) > 0 else 1
        s_out = classifier.classify(ch)
        if s_out and max(s_out.values()) > 0.0:
            raga_out = max(s_out, key=s_out.get)
        print_file_summary(duration_s, f_tonic, s_out, intended_raga, ch)
        deviation_report = compute_swara_deviation_report(ch, intended_raga)
    else:
        print("\nNo pitch could be detected anywhere in the file.")

    print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s=duration_s,
                         deviation_report=deviation_report, tonic_source=tonic_source, artist_name=artist_name)

def main():
    os.system("clear")
    print("=" * 60 + "\n         RAGADETECTOR TESTING SETUP LAB\n" + "=" * 60)
    print("(This label is only used to save/name the recording and score results")
    print(" afterward against the theoretical swar distribution - it is never given")
    print(" to the tonic or raga detector, which don't know the answer in advance.)")
    user_input = input("What Raaga do you plan to perform right now? (e.g. Marwa, Bhairav): ").strip()

    intended_raga = RAGA_SYNONYMS.get(user_input.lower(), user_input)
    if intended_raga not in RAGA_REGISTRY:
        # RAGA_SYNONYMS only covers a few known misspellings/nicknames - fall back
        # to a case-insensitive match against the registry itself, so a correctly
        # spelled but differently-cased name (e.g. "asavari" vs "Asavari") doesn't
        # error out unnecessarily.
        case_insensitive_match = next(
            (r for r in RAGA_REGISTRY if r.lower() == user_input.lower()), None)
        intended_raga = case_insensitive_match
    if intended_raga is None or intended_raga not in RAGA_REGISTRY:
        print(f"\n❌ ERROR: '{user_input}' is not in database registry!")
        print(f"Available choices: {list(RAGA_REGISTRY.keys())}\n")
        sys.exit(1)

    # Captured up front and saved into the recording's filename/JSON summary so a
    # known singer's identity doesn't have to be reconstructed by ear later - a
    # meaningful chunk of a recent session went into exactly that after-the-fact
    # forensic work, and it turned out several old recordings weren't even usable
    # classical performances at all (Bollywood songs, an interrupted broadcast).
    artist_input = input("Is the artist/singer known? Enter a name (e.g. 'Bhimsen Joshi', 'self'), "
                          "or press Enter if unknown: ").strip()
    artist_name = artist_input or None

    file_path = input("Path to an audio file to analyze (or press Enter to use the live mic): ").strip()

    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    classifier = RaagaClassifier(RAGA_REGISTRY)

    if file_path:
        if not os.path.isfile(file_path):
            print(f"\n❌ ERROR: file not found: {file_path}\n")
            sys.exit(1)
        tonic_input = input("Enter the documented/known tonic Sa frequency in Hz (e.g. 145), "
                             f"or press Enter to auto-detect from up to {int(LIVE_TONIC_WINDOW_SECONDS)} seconds of audio: ").strip()
        run_file_mode(file_path, tonic_input, extractor, classifier, intended_raga, artist_name=artist_name)
    else:
        # Live mode calculates its own tonic from the first LIVE_TONIC_WINDOW_SECONDS
        # of actual singing - no manual entry, no separate blind calibration window.
        run_live_mode(extractor, classifier, intended_raga, artist_name=artist_name)

if __name__ == "__main__":
    main()
