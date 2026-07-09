# -*- coding: utf-8 -*-
"""
7-Raga Studio Engine - Real-Time MIR Orchestrator
Features immediate dashboard ignition, correct configuration constants,
and seamless integration with FeatureExtractor for live tracking.
"""

import os
import numpy as np
import time
from collections import deque
import logging
import json
import sys
import librosa

from audio_stream import AudioStream
from feature_extraction import FeatureExtractor
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    HISTOGRAM_BINS,
    BINS_PER_SEMITONE,
    ROLLING_WINDOW_SECONDS,
    DECAY_FACTOR,
    RAAGA_DATABASE
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Standard upper-case swara index variables (same fixed order as config.SWARA_MAPPING)
SA, RE_K, RE, GA_K, GA, MA, MA_T, PA, DHA_K, DHA, NI_K, NI = range(12)
SWARA_NAMES = ['Sa', 'Re_K', 'Re', 'Ga_K', 'Ga', 'Ma', 'Ma_T', 'Pa', 'Dha_K', 'Dha', 'Ni_K', 'Ni']

RAGA_REGISTRY = RAAGA_DATABASE

RAGA_SYNONYMS = {"marva": "Marwa", "bhoop": "Bhupali", "bhoopali": "Bhupali", "bhairv": "Bhairav"}

TONIC_DETECTION_SECONDS = 5.0
FALLBACK_TONIC_HZ = 145.0

class RaagaClassifier:
    def __init__(self, db):
        self.db = db
    def classify(self, hist):
        cl = hist.copy(); cl[cl < 0.05] = 0.0
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
    print(f" Time Remaining   : {max(0, 120 - int(elapsed))} seconds | Active Tonic (Sa): {tonic} Hz\n" + "-" * 70)

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

def print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s):
    meta = RAGA_REGISTRY.get(raga_out, {"vadi": "Unknown", "samvadi": "Unknown", "pakad": []})
    is_match = "PASSED ✅" if raga_out.lower() == intended_raga.lower() else "FAILED ❌"
    print(f"\n============================================================\n CONCURRENT DETECTED RAAGA : {raga_out}\n VALIDATION LAB STATUS      : {is_match}\n------------------------------------------------------------")
    print(json.dumps({
        "session_metadata": {"execution_duration_seconds": round(duration_s, 2), "input_tonic_sa_hz": f_tonic},
        "classification_results": {"detected_dominant_raga": raga_out, "validation_status": is_match},
        "musicological_ground_truth": {
            "raga_name": raga_out,
            "vadi_king_note": meta["vadi"],
            "samvadi_queen_note": meta["samvadi"],
            "pakad_signature_phrase": " -> ".join(meta["pakad"])
        }
    }, indent=2))

def detect_tonic(extractor, chunks, fallback=FALLBACK_TONIC_HZ):
    """Auto-detect tonic from a batch of audio chunks, falling back to a default on failure."""
    if not chunks:
        print(f"No audio captured for tonic detection; using fallback {fallback} Hz")
        return fallback
    tonic = extractor.auto_detect_tonic(chunks, method='median')
    if tonic is None:
        print(f"Could not detect a tonic from the audio; using fallback {fallback} Hz")
        return fallback
    print(f"Auto-detected tonic: {tonic} Hz")
    return tonic

def load_file_chunks(filepath, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE):
    """Load an audio file and split it into BUFFER_SIZE-sized chunks, resampled to match the mic pipeline."""
    audio, _ = librosa.load(filepath, sr=sample_rate, mono=True)
    return [audio[i:i + buffer_size] for i in range(0, len(audio), buffer_size)]

def run_live_mode(tonic_input, extractor, classifier, intended_raga):
    streamer = AudioStream(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE)
    streamer.start()

    if tonic_input:
        try:
            f_tonic = float(tonic_input)
        except ValueError:
            f_tonic = FALLBACK_TONIC_HZ
    else:
        print(f"\nSing or hum your Sa now - listening for {TONIC_DETECTION_SECONDS:.0f} seconds to auto-detect tonic...")
        detect_chunks = []
        detect_start = time.time()
        while time.time() - detect_start < TONIC_DETECTION_SECONDS:
            while len(streamer.audio_queue) > 0:
                detect_chunks.append(streamer.audio_queue.popleft())
            time.sleep(0.05)
        f_tonic = detect_tonic(extractor, detect_chunks)

    extractor.set_tonic(f_tonic)
    history = deque(maxlen=int(ROLLING_WINDOW_SECONDS * (SAMPLE_RATE / BUFFER_SIZE)))
    start_time = time.time()
    raga_out = "Undetermined"
    s_out = {}

    # Pre-draw dashboard immediately on ignition - no blank screen
    empty_hist = np.zeros(HISTOGRAM_BINS)
    empty_scores = {k: 0.0 for k in RAGA_REGISTRY.keys()}
    draw_dashboard(0, f_tonic, empty_scores, intended_raga, empty_hist)

    try:
        while True:
            dt = time.time() - start_time
            if dt >= 120.0: break

            queue_size = len(streamer.audio_queue)
            if queue_size > 0:
                print(f"[DEBUG] Queue size: {queue_size}", end='\r')

            while len(streamer.audio_queue) > 0:
                c = streamer.audio_queue.popleft()
                p = extractor.extract_pitch(c)
                if p and not np.isnan(p):
                    try:
                        extractor.add_to_histogram(p)
                        frame_hist = extractor.get_normalized_histogram()
                        if frame_hist is not None and np.sum(frame_hist) > 0:
                            history.append(frame_hist)
                    except Exception as e:
                        pass

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
        print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s=120.0)

def run_file_mode(file_path, tonic_input, extractor, classifier, intended_raga):
    print(f"\nLoading audio file: {file_path}")
    chunks = load_file_chunks(file_path)
    if not chunks:
        print("\n❌ ERROR: could not read any audio from the file.\n")
        sys.exit(1)

    if tonic_input:
        try:
            f_tonic = float(tonic_input)
        except ValueError:
            f_tonic = FALLBACK_TONIC_HZ
        performance_chunks = chunks
    else:
        # Treat the opening TONIC_DETECTION_SECONDS as a calibration reference (e.g. a
        # held Sa), not performance content - folding it into the classification
        # histogram would let it dominate and bias the result toward whatever bin it
        # happens to land in.
        detect_chunk_count = int(TONIC_DETECTION_SECONDS * SAMPLE_RATE / BUFFER_SIZE)
        f_tonic = detect_tonic(extractor, chunks[:detect_chunk_count])
        performance_chunks = chunks[detect_chunk_count:] or chunks

    extractor.set_tonic(f_tonic)
    history = []
    for chunk in performance_chunks:
        p = extractor.extract_pitch(chunk)
        if p and not np.isnan(p):
            try:
                extractor.add_to_histogram(p)
                frame_hist = extractor.get_normalized_histogram()
                if frame_hist is not None and np.sum(frame_hist) > 0:
                    history.append(frame_hist)
            except Exception:
                pass

    duration_s = len(chunks) * BUFFER_SIZE / SAMPLE_RATE
    raga_out = "Undetermined"
    s_out = {}
    if history:
        ch = np.mean(history, axis=0)
        ch /= np.sum(ch) if np.sum(ch) > 0 else 1
        s_out = classifier.classify(ch)
        if s_out and max(s_out.values()) > 0.0:
            raga_out = max(s_out, key=s_out.get)
        print_file_summary(duration_s, f_tonic, s_out, intended_raga, ch)
    else:
        print("\nNo pitch could be detected anywhere in the file.")

    print_final_summary(raga_out, s_out, f_tonic, intended_raga, duration_s=duration_s)

def main():
    os.system("clear")
    print("=" * 60 + "\n         RAGADETECTOR TESTING SETUP LAB\n" + "=" * 60)
    user_input = input("What Raaga do you plan to perform right now? (e.g. Marwa, Bhairav): ").strip()

    intended_raga = RAGA_SYNONYMS.get(user_input.lower(), user_input)
    if intended_raga not in RAGA_REGISTRY:
        print(f"\n❌ ERROR: '{user_input}' is not in database registry!")
        print(f"Available choices: {list(RAGA_REGISTRY.keys())}\n")
        sys.exit(1)

    file_path = input("Path to an audio file to analyze (or press Enter to use the live mic): ").strip()

    tonic_input = input("Enter target Artist Tonic Sa frequency in Hz (e.g. 145), "
                         f"or press Enter to auto-detect from {TONIC_DETECTION_SECONDS:.0f} seconds of audio: ").strip()

    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
    classifier = RaagaClassifier(RAGA_REGISTRY)

    if file_path:
        if not os.path.isfile(file_path):
            print(f"\n❌ ERROR: file not found: {file_path}\n")
            sys.exit(1)
        run_file_mode(file_path, tonic_input, extractor, classifier, intended_raga)
    else:
        run_live_mode(tonic_input, extractor, classifier, intended_raga)

if __name__ == "__main__":
    main()
