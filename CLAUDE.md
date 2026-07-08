# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-only, real-time Hindustani classical Raaga (raga) identifier. It captures live microphone audio, extracts pitch, converts it to a tonic-relative 120-bin pitch-class histogram, and compares that histogram against reference Raaga templates via cosine similarity to guess which Raaga is being performed.

## Running

```bash
pip install -r requirements.txt
python3 main_22.py    # 7-raga "studio" mode: asks which raga you intend to sing, tracks a live 12-swara spectrum vs. the target, prints a PASS/FAIL validation at the end
python3 main.py        # 2-raga (Yaman/Bhupali) mode with tie-breaking/lock logic for a stuck classifier
```

Both entry points run for a fixed 120-second window, redraw a terminal dashboard once per second, and print a JSON summary on exit.

Module self-tests (no test framework/pytest is set up in this repo):
```bash
python3 config.py              # validates RAAGA_DATABASE histograms via validate_config()
python3 audio_stream.py        # opens the mic for 5s and prints RMS/queue stats
python3 feature_extraction.py  # runs a synthetic 440Hz sine wave through the full pitch->bin pipeline
```

Note: `requirements.txt` is unpinned to Python 3.8-era versions (numpy 1.21, librosa 0.9), but `instructions.md` (the original design brief, see below) specifies stricter pins (`numpy<1.20`, etc.) — the installed requirements.txt takes precedence over instructions.md when they disagree.

## Architecture

### Pipeline shape

`audio_stream.AudioStream` (PyAudio callback thread) → raw float32 chunks pushed into a `deque(maxlen=100)` → consumer pulls chunks → `feature_extraction.FeatureExtractor` does autocorrelation pitch detection → converts Hz to cents relative to a fixed tonic (`1200 * log2(f/f_tonic)`) → wraps to a single octave (0–1200 cents) → bins into one of 120 histogram bins (10 bins/semitone, for shruti-level resolution) → a rolling `deque` of recent frame-histograms is averaged into a "current" histogram → a classifier does cosine similarity against per-Raaga reference histograms → scores are min-max normalized and the argmax is the detected Raaga.

Tonic (Sa) is **not auto-detected in the running apps** — both `main.py` and `main_22.py` prompt the user for the tonic frequency in Hz at startup and pass it straight into `FeatureExtractor(tonic_frequency=...)`. `FeatureExtractor.auto_detect_tonic()` exists but is unused by either entry point.

### The three Raaga databases are independent, not shared

This is the most important non-obvious thing about the codebase: there are **three separate, hand-maintained Raaga definitions** that have drifted apart, not one shared source of truth:

- `config.py::RAAGA_DATABASE` — the "rich" schema (aroha, avroha, pakad, vadi, samvadi, prahar, forbidden/required note masks, histogram built via `create_swara_histogram`). **Not imported or used by either `main.py` or `main_22.py`.**
- `main.py::RAGA_REGISTRY` — 2 ragas (Yaman, Bhupali), histograms built inline via a local `generate_idealized_histogram()` (Gaussian-smoothed weights per semitone), plus ad hoc classifier hacks (e.g. a hardcoded Yaman similarity bonus when high-Ni energy is present, and a `check_tie()` lock mechanism that force-locks onto "Bhupali" after 90s if Ma/Ni energy stays low).
- `main_22.py::RAGA_REGISTRY` — 7 ragas (Yaman, Bhupali, Bhairav, Asavari, Jaunpuri, Marwa, Puriya), also built via a local (differently-named but similarly-shaped) `generate_idealized_histogram()`, plus a `forbidden` semitone list per raga that applies a flat 0.15 similarity penalty if forbidden-note energy exceeds 4%.

When adding/editing a Raaga or tuning classification behavior, check **which file's classifier you're actually running** — changes to `config.py` currently have no effect on either app's live behavior.

### Key numeric conventions

- 120-bin histogram = 12 semitones × 10 bins/semitone (`BINS_PER_SEMITONE`), giving ~10 cents/bin resolution for microtonal (shruti) analysis.
- Semitone index ordering is fixed: `Sa=0, Re_komal=1, Re_shuddha=2, Ga_komal=3, Ga_shuddha=4, Ma_shuddha=5, Ma_tivra=6, Pa=7, Dha_komal=8, Dha_shuddha=9, Ni_komal=10, Ni_shuddha=11` (see `config.SWARA_MAPPING`; `main_22.py` redefines the same order locally as `SA, RE_K, RE, GA_K, GA, MA, MA_T, PA, DHA_K, DHA, NI_K, NI`).
- `main.py` and `main_22.py` each hardcode their own `SAMPLE_RATE`/`HISTOGRAM_BINS`/etc. rather than importing all of them from `config.py` — `main.py` in particular overrides `SAMPLE_RATE` to 22050 locally instead of using `config.SAMPLE_RATE` (44100).

### Threading/queue model

`AudioStream` uses a PyAudio non-blocking stream with a callback (`_audio_callback`) that computes RMS, adapts a silence threshold (exponential toward current RMS, clamped to `[min_rms_threshold, max_rms_threshold]`), and pushes non-silent chunks onto `self.audio_queue` (a plain `collections.deque`, not `queue.Queue` — not inherently thread-safe against concurrent mutation, relies on GIL + single-consumer usage). The `_capture_loop` thread itself does nothing but sleep; all real work happens in the PyAudio callback, which runs on PyAudio's own internal thread.

### instructions.md

`instructions.md` is the original system-prompt/design brief this project was built from (aspirational scope: HMMs, SVMs, TensorFlow Lite, tala detection, etc.). Treat it as background/intent, not as a description of what's actually implemented — most of its advanced ML/architecture sections (plugin architecture, HMMs, ensemble classifiers, Cython/Numba) are not present in the code.
