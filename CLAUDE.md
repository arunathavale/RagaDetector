# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-only Hindustani classical Raaga (raga) identifier that works from either live microphone audio or a dropped-in audio file. It extracts pitch, converts it to a tonic-relative 120-bin pitch-class histogram, and compares that histogram against reference Raaga templates via cosine similarity to guess which Raaga is being performed.

## Running

```bash
pip install -r requirements.txt
python3 main_22.py    # 7-raga "studio" mode: asks which raga you intend to sing, tracks a 12-swara spectrum vs. the target, prints a PASS/FAIL validation at the end
```

`main_22.py` is the sole entry point (the old 2-raga `main.py` was retired in Phase 4 — see `PROJECT_PLAN.md`). It prompts for:
1. The raga you intend to perform. **This is a label only** — used to name/save the recording and to score the PASS/FAIL result and theoretical-vs-actual swar comparison after the fact. It is never given to the tonic or raga detector, which don't know the answer in advance — the whole point of the project is to identify an raga that's genuinely unknown to the system. (An earlier version of the live-mode tonic redesign below violated this by searching for "the tonic that makes the intended raga classify best," which is circular — it only validated correctly because the true answer happened to be supplied as the label. Fixed by switching to a raga-agnostic method.)
2. An audio file path, or Enter for live mic.
3. **File mode only**: a tonic (Sa) frequency in Hz, or Enter to auto-detect from the opening ~5 seconds of audio (`FeatureExtractor.auto_detect_tonic()`, median method), which is treated as a calibration reference and excluded from the classification histogram.

**Live mic mode no longer asks for a tonic at all.** Manual entry and a separate blind "sing your Sa for 5 seconds" window were both tested extensively and repeatedly proved wrong or misleading — a live dashboard built on either is not trustworthy. Instead, `run_live_mode()` runs in two phases against one fixed `SESSION_DURATION_SECONDS` (4 min) budget:
- **Phase 1** (`LIVE_TONIC_WINDOW_SECONDS`, 2 min): listens silently — no dashboard — while the singer performs normally. Two separate pitch streams are built from the same audio: per-chunk autocorrelation (`extractor.extract_pitch()`, one call per `BUFFER_SIZE` chunk as it arrives — same method Phase 2 uses for live tracking) supplies the swara/deviation-report content, while `continuous_pyin_pitches()` runs `librosa.pyin()` **once**, continuously, across the *whole* window with a proper ~10ms hop at the end of Phase 1 and supplies the pitches used for the tonic decision specifically. These are deliberately different: pyin was ruled out for Phase 2's live swara tracking earlier this session for being source-agnostic (confident about any stable pitch, not specifically the voice) — a liability there, but the right bias for isolating a stable tanpura drone, which is exactly why it's used here instead. Comparison of 5 candidate tonic methods (`compare_tonic_methods.py`, against all 10 saved recordings) found continuous pyin beat plain per-chunk autocorrelation by ~35% on average, clearly ahead of stability-filtering, HPSS preprocessing, or combining pyin with stability-filtering — see `PROJECT_PLAN.md`'s 2026-07-10 entries. `determine_tonic_from_pitches()` then finds the tonic via **drone-interval priors** over whichever pitch list it's given — a tanpura almost always drones on Sa+Pa (fifth) or Sa+Ma (fourth), a raga-agnostic structural signal from how the accompanying instrument is tuned, not from what's being sung or which raga it turns out to be. Same technique as `find_optimal_tonic.py`'s `score_drone_priors()`/`find_best_tonic_by_drone_priors()`, reimplemented locally to avoid a circular import. `print_tonic_stability_check()` then splits the tonic-window pitches into 4 time-ordered segments and re-runs the same drone-prior search on each, printing the spread (skipped if there are too few pitches to split meaningfully — pyin's confidence threshold is far more selective than autocorrelation, sometimes only a few dozen detections in a 2-minute window) — tonic essentially never changes mid-performance, so disagreement across segments flags a pitch-tracking problem rather than a real shift (same diagnostic as `find_optimal_tonic.py`'s `check_tonic_stability()`).
- **Phase 2** (remaining 2 min): the normal live dashboard, using the tonic calculated in Phase 1.

Phase 1's pitch detections are real performance content, not throwaway calibration — they're converted to histogram frames (once tonic is known) and folded into the session's `history`/deviation report alongside Phase 2's, unlike the old calibration window which was excluded from both file mode and the pre-redesign live mode.

Drone-prior tonic detection is a real, working, complementary diagnostic but not a fully solved problem — on the best-available ground-truth test this session it landed within ~1-2 semitones of the classifier-validated answer, good enough to be clearly better than a blind guess but not perfect. Treat a large stability-check spread as a genuine warning sign, not noise to ignore.

Live mode redraws a terminal dashboard once per second during Phase 2, and prints a JSON summary on exit. File mode processes the whole file once and prints a one-shot summary plus the same JSON block — no artificial time cap, since a file already has a finite, known length.

Every live mic session's raw audio (calibration + performance) is saved to `recordings/{raga}_{timestamp}.wav` when the session ends, regardless of pass/fail — live audio was previously processed and discarded in real time with no way to replay a session afterward. Saved recordings load through the same `load_file_chunks()` file mode uses, so any past session can be re-run through file mode directly (e.g. to compare pitch-detection methods, or as source material for empirical reference histograms). `recordings/` is gitignored.

Auto-detected tonic is only as accurate as the underlying autocorrelation pitch detector's precision (median method) — in testing it landed within ~30 cents of the true tonic on a clean synthetic tone, which was enough on its own to occasionally push a borderline case toward an adjacent raga. This is a known precision ceiling of the existing algorithm, not something this wiring changed; a manually-entered tonic is more reliable when accuracy matters.

`FeatureExtractor._pitch_autocorrelation()` (`feature_extraction.py`) picks the tallest autocorrelation peak within the valid `[min_freq, max_freq]` lag range and requires it to clear a minimum confidence (30% of the zero-lag value) - not simply the first peak found scanning outward from lag 0. The naive first-peak version was found to be prone to octave/harmonic errors on real (non-synthetic) audio: on one real recording, 64% of detected "pitches" were an implausible >600Hz, and about half of those had a taller, more plausible peak sitting at a longer lag that the old logic never examined. `eval_harness.py`'s clean sine-tone synthesis never exposed this, since a pure tone's first and tallest peaks coincide. See `PROJECT_PLAN.md`'s 2026-07-10 entries for the full investigation, including a regression this fix's first draft caused in `eval_harness.py` (95.2% → 74.6%, traced to an unrelated one-sample peak-index change bundled into the same edit, not the octave-error fix itself) and how it was isolated and reverted.

Module self-tests (no test framework/pytest is set up in this repo):
```bash
python3 config.py              # validates RAAGA_DATABASE histograms via validate_config()
python3 audio_stream.py        # opens the mic for 5s and prints RMS/queue stats
python3 feature_extraction.py  # runs a synthetic 440Hz sine wave through the full pitch->bin pipeline
python3 eval_harness.py        # no mic needed: synthesizes each raga's aroha/avroha/pakad as sine tones,
                                # runs them through the real pitch pipeline, and scores classifier accuracy
```

Note: `requirements.txt` is unpinned to Python 3.8-era versions (numpy 1.21, librosa 0.9), but `instructions.md` (the original design brief, see below) specifies stricter pins (`numpy<1.20`, etc.) — the installed requirements.txt takes precedence over instructions.md when they disagree. **If anything in this environment starts throwing unexpected import errors** (e.g. `numpy has no attribute 'long'` from librosa/numba), check `pip show numpy scipy scikit-learn librosa` against `requirements.txt` before debugging further — a partially-failed `pip install` of an unrelated package has silently clobbered these pins before.

## Architecture

### Pipeline shape

`audio_stream.AudioStream` (PyAudio callback thread) → raw float32 chunks pushed into a `deque(maxlen=100)` → consumer pulls chunks → `feature_extraction.FeatureExtractor` does autocorrelation pitch detection → converts Hz to cents relative to a fixed tonic (`1200 * log2(f/f_tonic)`) → wraps to a single octave (0–1200 cents) → bins into one of 120 histogram bins (10 bins/semitone, for shruti-level resolution) → a rolling `deque` of recent frame-histograms is averaged into a "current" histogram → `RaagaClassifier.classify()` does plain cosine similarity against per-Raaga reference histograms → scores are min-max normalized and the argmax is the detected Raaga.

Tonic (Sa) handling differs by mode. File mode can be typed in manually or auto-detected (Phase 5, see `PROJECT_PLAN.md`) — `main_22.py` prompts for a Hz value and, if left blank, calls `FeatureExtractor.auto_detect_tonic()` against a calibration window of audio. Live mode calculates it automatically via `determine_tonic_from_pitches()` (see "Running" above) — no prompt at all. Both paths end by calling `extractor.set_tonic()` before the actual performance/classification.

### One Raaga database, one entry point

`config.py::RAAGA_DATABASE` is the single source of truth for all 7 ragas (Yaman, Bhupali, Bhairav, Asavari, Jaunpuri, Marwa, Puriya) — aroha, avroha, pakad, vadi, samvadi, prahar, forbidden/required notes, and a `histogram` field. `main_22.py` imports `RAAGA_DATABASE` directly; there is no separate inline registry anymore.

Each raga's `histogram` is built by `config.compute_swara_weights()` + `create_weighted_swara_histogram()` in a post-construction pass over `RAAGA_DATABASE`: each swara is weighted by how often it occurs across that raga's own aroha+avroha+pakad, with an extra boost for the vadi/samvadi notes, then normalized. This matters because some raga pairs share an identical note set (e.g. Asavari/Jaunpuri, Marwa/Puriya, same thaat) — a flat "note present/absent" histogram makes those pairs literally indistinguishable (cosine similarity 1.0000); the weighting scheme differentiates them by melodic emphasis instead. See `PROJECT_PLAN.md`'s Phase 2/3 session log for the before/after numbers.

`RaagaClassifier.classify()` in `main_22.py` is plain cosine similarity with **no hand-tuned bonuses, penalties, or lock heuristics** — Phase 3 ablation-tested the old forbidden-note penalty and (in the now-deleted `main.py`) a Yaman similarity bonus and a 90-second tie-break lock; all three measured zero accuracy benefit on the eval harness once the weighted histograms were in place, so they were removed rather than kept as unverified complexity.

### Eval harness

`eval_harness.py` is the answer to "does this change actually help?" — it synthesizes sine-tone renditions of each raga's aroha/avroha/pakad across multiple tonics and multiple noisy trials (random per-note cents jitter), runs them through the real `FeatureExtractor` pipeline (no mic needed), classifies them, and reports accuracy plus a confusion matrix. Run it after any change to `config.py`'s histograms or `main_22.py`'s classifier logic before deciding the change "worked" — this is the project's documented lesson from a period of ungrounded, symptom-chasing fixes (see `PROJECT_PLAN.md` section 3).

### Key numeric conventions

- 120-bin histogram = 12 semitones × 10 bins/semitone (`BINS_PER_SEMITONE`), giving ~10 cents/bin resolution for microtonal (shruti) analysis.
- Semitone index ordering is fixed: `Sa=0, Re_komal=1, Re_shuddha=2, Ga_komal=3, Ga_shuddha=4, Ma_shuddha=5, Ma_tivra=6, Pa=7, Dha_komal=8, Dha_shuddha=9, Ni_komal=10, Ni_shuddha=11` (see `config.SWARA_MAPPING`; `main_22.py` redefines the same order locally as `SA, RE_K, RE, GA_K, GA, MA, MA_T, PA, DHA_K, DHA, NI_K, NI`).
- `main_22.py` hardcodes its own `SAMPLE_RATE`/`HISTOGRAM_BINS`/etc. via import from `config.py` rather than redefining them locally (this was cleaned up in Phase 1 — it used to hardcode its own copies).

### Threading/queue model

`AudioStream` uses a PyAudio non-blocking stream with a callback (`_audio_callback`) that computes RMS, adapts a silence threshold (exponential toward current RMS, clamped to `[min_rms_threshold, max_rms_threshold]`), and pushes non-silent chunks onto `self.audio_queue` (a plain `collections.deque`, not `queue.Queue` — not inherently thread-safe against concurrent mutation, relies on GIL + single-consumer usage). The `_capture_loop` thread itself does nothing but sleep; all real work happens in the PyAudio callback, which runs on PyAudio's own internal thread.

### instructions.md

`instructions.md` is the original system-prompt/design brief this project was built from (aspirational scope: HMMs, SVMs, TensorFlow Lite, tala detection, etc.). Treat it as background/intent, not as a description of what's actually implemented — most of its advanced ML/architecture sections (plugin architecture, HMMs, ensemble classifiers, Cython/Numba) are not present in the code.
