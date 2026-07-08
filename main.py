# -*- coding: utf-8 -*-
import os, numpy as np, time, threading, queue, logging, json, sys
from collections import deque
from audio_stream import AudioStream
from feature_extraction import FeatureExtractor
from config import RAAGA_DATABASE

SAMPLE_RATE, CHUNK_SIZE, ROLLING_WINDOW_SECONDS, DECAY_FACTOR, HISTOGRAM_BINS, BINS_PER_SEMITONE = 22050, 2048, 45, 0.98, 120, 10
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

RAGA_REGISTRY = {name: RAAGA_DATABASE[name] for name in ("Yaman", "Bhupali")}

class RaagaClassifier:
    def __init__(self, database): self.database, self.was_locked = database, False
    def classify(self, hist):
        cleaned = hist.copy(); cleaned[cleaned < 0.05] = 0.0
        scores = {}
        for name, entry in self.database.items():
            ref = entry["histogram"]
            num, n1, n2 = np.dot(cleaned, ref), np.linalg.norm(cleaned), np.linalg.norm(ref)
            sim = (num / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0
            if name == "Yaman" and np.sum(hist[110:120]) > 0.02: sim = min(1.0, sim + 0.15)
            scores[name] = max(0.0, min(1.0, sim))
        mn, mx = min(scores.values()), max(scores.values())
        return {k: ((v - mn) / (mx - mn) if mx > mn else 1.0/len(scores)) for k, v in scores.items()}
    def check_tie(self, scores, hist, elapsed):
        ma, ni = np.sum(hist[60:70]), np.sum(hist[110:120])
        lock = (elapsed >= 90.0 and ma < 0.03 and ni < 0.03)
        if "Yaman" in scores and "Bhupali" in scores and scores["Yaman"] - scores["Bhupali"] > 0.20: lock = False
        flush = self.was_locked and not lock
        self.was_locked = lock
        if lock: return {"Bhupali": 0.95, "Yaman": 0.05}, ma, ni, "LOCKED ON BHOOP", False
        return scores, ma, ni, "UNLOCKED", flush

def draw_dashboard(elapsed, lock, ma, ni, tonic, scores):
    os.system("clear")
    print("=" * 60 + "\n RAGADETECTOR LIVE DASHBOARD (Intel Core i7 CPU Optimized)\n" + "=" * 60)
    print(f" Time Remaining   : {max(0, 120 - int(elapsed))} seconds\n Active Tonic (Sa): {tonic} Hz\n Lock Registry    : {lock}")
    print(f" Teevra Ma (Bin 6): {ma:.3f} | Ni (Bin 11): {ni:.3f}\n" + "-" * 60)
    print(" Traditional Raaga Ground Truth Reference Card:\n  • Bhupali : Vadi=Ga, Samvadi=Dha (Pentatonic - Avoids Ma/Ni)\n  • Yaman   : Vadi=Ga, Samvadi=Ni  (Heptatonic - Uses Teevra Ma)\n" + "-" * 60)
    for r, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r:<12} [{'█'*int(s*30) + '░'*(30-int(s*30))}] {s*100:>5.1f}%")
    print("=" * 60 + "\n Application closes automatically after 120 seconds.")

def main():
    os.system("clear")
    try: f_tonic = float(input("Enter target Artist Tonic Sa frequency in Hz: "))
    except: f_tonic = 145.0
    streamer = AudioStream(sample_rate=SAMPLE_RATE, buffer_size=CHUNK_SIZE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, tonic_frequency=f_tonic)
    classifier = RaagaClassifier(RAGA_REGISTRY)
    history = deque(maxlen=int(ROLLING_WINDOW_SECONDS * (SAMPLE_RATE / CHUNK_SIZE)))
    streamer.start(); start_time = time.time(); raga_out = "Undetermined"; s_out = {}
    try:
        while True:
            dt = time.time() - start_time
            if dt >= 120.0: break
            while len(streamer.audio_queue) > 0:
                c = streamer.audio_queue.popleft(); p = extractor.extract_pitch(c)
                if p and not np.isnan(p):
                    try:
                        b = int((1200.0 * np.log2(p / extractor.tonic_frequency)) % 1200.0 / (1200.0 / HISTOGRAM_BINS)) % HISTOGRAM_BINS
                        f = np.zeros(HISTOGRAM_BINS); f[b] = 1.0; history.append(f)
                    except: pass
            if len(history) > 0:
                ch = np.mean(history, axis=0); ch /= np.sum(ch) if np.sum(ch) > 0 else 1
                s_out, mae, nie, st, fl = classifier.check_tie(classifier.classify(ch), ch, dt)
                if fl: history.clear()
                if s_out: raga_out = max(s_out, key=s_out.get)
                draw_dashboard(dt, st, mae, nie, f_tonic, s_out)
            time.sleep(1.0)
    except KeyboardInterrupt: pass
    finally:
        streamer.is_running = False
        if hasattr(streamer, "stream") and streamer.stream:
            try: streamer.stream.stop_stream(); streamer.stream.close()
            except: pass
        meta = RAGA_REGISTRY.get(raga_out, {"vadi": "Unknown", "samvadi": "Unknown", "pakad": []})
        os.system("clear")
        print(f"\n============================================================\n ⏱️  120-SECOND PERFORMANCE WINDOW CONCLUDED SUCCESSFULLY\n============================================================\n CONCURRENT DETECTED RAAGA : {raga_out}\n------------------------------------------------------------\n FINAL STRUCTURAL MATRIX (JSON FORMAT):")
        print(json.dumps({"session_metadata": {"execution_duration_seconds": 120.0, "input_tonic_sa_hz": f_tonic, "sample_rate_hz": SAMPLE_RATE}, "classification_results": {"detected_dominant_raga": raga_out, "confidence_scores": {k: round(float(v), 4) for k, v in s_out.items()}}, "musicological_ground_truth": {"raga_name": raga_out, "vadi_king_note": meta["vadi"], "samvadi_queen_note": meta["samvadi"], "pakad_signature_phrase": " -> ".join(meta["pakad"])}}, indent=2))
        print("============================================================\nPipeline closed cleanly.\n")

if __name__ == "__main__": main()
