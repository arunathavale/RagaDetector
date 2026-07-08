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

from audio_stream import AudioStream
from feature_extraction import FeatureExtractor
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    HISTOGRAM_BINS,
    BINS_PER_SEMITONE,
    ROLLING_WINDOW_SECONDS,
    DECAY_FACTOR,
    SWARA_MAPPING,
    RAAGA_DATABASE
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Standard upper-case swara index variables (same fixed order as config.SWARA_MAPPING)
SA, RE_K, RE, GA_K, GA, MA, MA_T, PA, DHA_K, DHA, NI_K, NI = range(12)
SWARA_NAMES = ['Sa', 'Re_K', 'Re', 'Ga_K', 'Ga', 'Ma', 'Ma_T', 'Pa', 'Dha_K', 'Dha', 'Ni_K', 'Ni']

RAGA_REGISTRY = RAAGA_DATABASE

RAGA_SYNONYMS = {"marva": "Marwa", "bhoop": "Bhupali", "bhoopali": "Bhupali", "bhairv": "Bhairav"}

class RaagaClassifier:
    def __init__(self, db):
        self.db = db
        self.forbidden_indices = {
            name: [SWARA_MAPPING[n] for n in entry["forbidden_notes"]]
            for name, entry in db.items()
        }
    def classify(self, hist):
        cl = hist.copy(); cl[cl < 0.05] = 0.0
        scores = {}
        for name, entry in self.db.items():
            penalty = 1.0
            if any(np.sum(cl[i*BINS_PER_SEMITONE:(i+1)*BINS_PER_SEMITONE]) > 0.04 for i in self.forbidden_indices[name]):
                penalty = 0.15
            ref = entry["histogram"]
            num, n1, n2 = np.dot(cl, ref), np.linalg.norm(cl), np.linalg.norm(ref)
            sim = (num / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0
            scores[name] = max(0.0, min(1.0, sim * penalty))
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

def main():
    os.system("clear")
    print("=" * 60 + "\n         RAGADETECTOR TESTING SETUP LAB\n" + "=" * 60)
    user_input = input("What Raaga do you plan to perform right now? (e.g. Marwa, Bhairav): ").strip()
    
    intended_raga = RAGA_SYNONYMS.get(user_input.lower(), user_input)
    if intended_raga not in RAGA_REGISTRY:
        print(f"\n❌ ERROR: '{user_input}' is not in database registry!")
        print(f"Available choices: {list(RAGA_REGISTRY.keys())}\n")
        sys.exit(1)
        
    try: f_tonic = float(input("Enter target Artist Tonic Sa frequency in Hz (e.g. 145): "))
    except: f_tonic = 145.0
    
    streamer = AudioStream(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, tonic_frequency=f_tonic)
    classifier = RaagaClassifier(RAGA_REGISTRY)
    history = deque(maxlen=int(ROLLING_WINDOW_SECONDS * (SAMPLE_RATE / BUFFER_SIZE)))
    streamer.start()
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
        meta = RAGA_REGISTRY.get(raga_out, {"vadi": "Unknown", "samvadi": "Unknown", "pakad": []})
        is_match = "PASSED ✅" if raga_out.lower() == intended_raga.lower() else "FAILED ❌"
        os.system("clear")
        print(f"\n============================================================\n CONCURRENT DETECTED RAAGA : {raga_out}\n VALIDATION LAB STATUS      : {is_match}\n------------------------------------------------------------")
        print(json.dumps({"session_metadata": {"execution_duration_seconds": 120.0, "input_tonic_sa_hz": f_tonic}, "classification_results": {"detected_dominant_raga": raga_out, "validation_status": is_match}, "musicological_ground_truth": {"raga_name": raga_out, "vadi_king_note": meta["vadi"], "samvadi_queen_note": meta["samvadi"], "pakad_signature_phrase": " -> ".join(meta["pakad"])}}, indent=2))

if __name__ == "__main__":
    main()
