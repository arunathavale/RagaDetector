# RagaDetector — Project Plan

Last updated: 2026-07-08

This file is meant to be the anchor document for this project — read it at the start of any Claude Code session (alongside `CLAUDE.md`) before making changes, and update it as milestones are hit or the plan changes.

## 1. What this project is

A CPU-only, real-time Hindustani classical Raaga identifier. It listens to live microphone audio (or could take a dropped-in music file), extracts pitch, builds a tonic-relative 120-bin pitch-class histogram, and compares it against reference Raaga templates via cosine similarity to guess which Raaga is being performed.

## 2. Where things actually stand today

Two working prototype entry points exist, and they do run:

- `main.py` — a 2-raga mode (Yaman, Bhupali) with extra tie-breaking/"lock" logic for a classifier that tends to get stuck.
- `main_22.py` — a 7-raga "studio" mode (Yaman, Bhupali, Bhairav, Asavari, Jaunpuri, Marwa, Puriya) where you tell it which raga you intend to sing, and it shows a live PASS/FAIL comparison against that target.

Both run for a fixed 120-second window and print a JSON summary at the end. This is useful for testing, but it's a test-harness shape, not yet a "just detect whatever raga I'm singing" tool.

The honest problems, found while reviewing the code just now:

- ~~**Three separate, drifted Raaga databases.**~~ **Resolved in Phase 1 (2026-07-08).** `config.py::RAAGA_DATABASE` (the richest schema — aroha, avroha, pakad, vadi, samvadi, prahar) now covers all 7 ragas and is the single source both `main.py` and `main_22.py` import from; the inline registries and idealized-histogram generators were deleted from both.
- **Hand-tuned, per-raga hacks instead of a general classifier.** `main.py` has a hardcoded "Yaman similarity bonus" when high-Ni energy is present, and a `check_tie()` mechanism that force-locks onto "Bhupali" after 90 seconds under certain conditions. `main_22.py` applies a flat 0.15 similarity penalty when "forbidden note" energy exceeds an arbitrary 4% threshold. These read like patches added reactively to fix specific failed test runs, rather than a principled feature or model that generalizes.
- **Tonic (Sa) is entered by hand every run.** `FeatureExtractor.auto_detect_tonic()` exists in the code but isn't called by either entry point — you have to type in the Hz value yourself each time.
- **No objective evaluation.** There's no labeled dataset or test set to score classifier accuracy against. Progress so far has likely been judged by singing live and eyeballing the dashboard, which makes it very hard to tell whether a change actually helped or just fixed the one case you were staring at (this is almost certainly why work with Devin/Gemini started "spinning wheels" — every fix was a one-off reaction, with nothing to check it against, so old cases could quietly break while a new one got patched).
- **`instructions.md` is aspirational, not a status report.** It describes HMMs, SVMs, ensemble classifiers, tala detection, TensorFlow Lite, a plugin architecture — none of which exist in the code. It's useful as a reference for future direction, but it doesn't reflect what's built.
- **Housekeeping debt.** `raga_detector.log` had grown to 265KB and was being tracked by git before we added `.gitignore`; `__pycache__` likewise. Two entry points with overlapping but diverging logic is itself a maintenance cost.

None of this means the project is in bad shape — the core pipeline (pitch detection → cents conversion → histogram → cosine similarity) is a legitimate, well-understood approach for this problem, and it's already running end-to-end on live audio. The issue is architectural drift and a lack of a feedback loop to measure progress, not a fundamentally broken idea.

## 3. Why progress stalled

Most likely pattern: each AI coding session (Devin, Gemini) was handed a live symptom ("it keeps guessing Bhupali," "Yaman never gets picked") and asked to fix it, with no persistent ground-truth test set and no single source of truth for the raga definitions. Each tool fixed the symptom in front of it by adding a special case to whichever file it happened to be looking at, which is how you end up with three databases and several hardcoded bonuses/penalties. Without an eval set, there was no way to tell if a "fix" was a real improvement or an overfit to one test.

## 4. Recommended approach: fix the process, then the code

The methodical fix is to stop patching symptoms and put a few structural things in place first, in this order. Each phase is small enough to be one or two focused Claude Code sessions.

### Phase 1 — Consolidate to one source of truth
Pick a single Raaga database schema (recommend `config.py::RAAGA_DATABASE`'s richer schema — aroha, avroha, pakad, vadi, samvadi, forbidden/required notes — since it's the most complete) and make both entry points import from it. Delete the inline registries in `main.py` and `main_22.py`. Expand it to cover the 7 ragas `main_22.py` currently has inline, at minimum.

### Phase 2 — Build a real evaluation harness
Before touching classifier logic again, create a small labeled test set — even a handful of short recordings (or synthetic tone sequences generated from each raga's aroha/avroha) with known correct answers — and a script that runs the classifier against all of them and reports accuracy. This turns "does this help?" from a guess into a number you can compare run to run. This is the single highest-leverage step to stop the spinning-wheels pattern.

### Phase 3 — Replace ad hoc hacks with justified logic
With the eval harness in place, revisit the hardcoded bonuses/penalties/lock heuristics. Either remove them and see how much accuracy the plain cosine-similarity approach gets on its own, or keep only the ones that measurably improve the eval score — and document why each one exists.

**Two-stage classification (Arun's addition, 2026-07-08):** Turn "forbidden notes" from a soft similarity penalty into a first-pass elimination filter — if a raga's forbidden swara shows sustained energy in the live histogram above a threshold, drop that raga from the candidate set entirely before running cosine similarity on whatever remains. This is musicologically sound (several ragas are defined partly by which notes they exclude, e.g. Bhupali has no Ma/Ni) and should narrow the field faster than a penalty multiplier does. Caveat to test against the eval harness: the elimination threshold needs to tolerate pitch-detection noise (a single stray frame misreading a forbidden note shouldn't disqualify the correct raga) — likely wants the same kind of sustained/energy-over-time threshold the current 4% penalty check uses, not a one-frame trigger.

### Phase 4 — Retire `main.py`, keep one entry point
Arun's read (and it holds up): `main.py`'s 2-raga registry (Yaman, Bhupali) is a strict subset of `main_22.py`'s 7-raga registry, and `main_22.py` is the more recently edited file. `main.py`'s only unique content is its tie-break/lock hack, which Phase 3 should be evaluating (and likely discarding or replacing) anyway. Plan: once Phase 1–3 are done and `main_22.py` is confirmed stable, delete `main.py` rather than merging it — there's nothing in it worth preserving beyond what the eval harness already tells you about the lock heuristic.

### Phase 5 — Quality-of-life
Wire up `auto_detect_tonic()` so tonic doesn't need to be typed in by hand each run. Add support for dropping in a music file (not just live mic) as an input source, since that's part of the original goal.

### Phase 6 — Only now, consider the aspirational features
Things like pakad (phrase) matching, HMMs, ensemble classifiers, tala detection from `instructions.md` are legitimate v2 ideas, but only worth adding once there's a reliable way (the eval harness) to tell whether they actually improve results.

### Phase 7 — Practice/learning mode (Arun's addition, 2026-07-08)
Extend the existing "studio" comparison (already in `main_22.py`: pick a raga, sing, see live vs. target) into an actual learning tool that points out specific variations rather than just a coarse bar chart and PASS/FAIL. Depends on Phase 1 (needs the richer `config.py` schema — aroha/avroha/pakad, not just a flat histogram) and ideally Phase 2's eval harness for validating that the feedback given is musically correct. Not a separate project — it reuses the entire audio/pitch/histogram/database pipeline; only the "what do we do with the comparison" step differs from guess-the-raga mode. Candidate feedback types: per-swara sharp/flat deviation in cents (needs finer resolution than the current 12-bucket semitone sum), accidental forbidden-note hits, aroha/avroha sequence adherence.

### Phase 8 — Sequential/melodic movement analysis: Aroha/Avroha, pakad (Arun's addition, 2026-07-08)
The current histogram approach is order-blind — it only measures how much time is spent on each note, not what sequence they occur in. This means ragas sharing the same note set but distinguished by melodic movement (aroha/avroha shape, vadi/samvadi emphasis, pakad) can be indistinguishable to a pure histogram classifier, and it's why "static" feedback in Phase 7's learning mode could only ever confirm note choice, never direction/order (e.g. it can't tell Sa-Re-Ga apart from Sa-Ga-Re — same histogram, wrong movement). This is not new scope — `config.py`'s schema already has unused `aroha`/`avroha`/`pakad`/`characteristic_phrases` fields, and it's what `instructions.md` originally called for.

Requires new building blocks beyond what exists today: turning the continuous pitch stream into discrete note events (onset/offset segmentation — the pipeline currently only produces frame-by-frame pitch, not "note X held from time A to B"), then matching that note sequence against each raga's aroha/avroha/pakad (start with something simple like n-gram or edit-distance matching before reaching for HMMs, which is what `instructions.md` originally suggested). Ornamentation (meend/gamak) will make segmentation noisy — expect this to need its own tuning pass. Depends on Phase 1 (schema) and should be validated against Phase 2's eval harness rather than by ear, same lesson as the rest of the classifier work. Feeds directly into Phase 7's learning-mode feedback once it exists.

## 5. Working style going forward (to avoid repeating the stall)

- Treat this file and `CLAUDE.md` as the two things to read at the start of any session.
- Tackle one phase at a time; don't let a session wander across phases.
- After any change to classifier logic, run it against the eval harness (once Phase 2 exists) before deciding it "worked."
- Commit and push after each phase (or meaningful sub-step), with a commit message describing what changed and why — not just "fix."
- Update this file's status section (or add a dated log entry below) at the end of each session so the next session — or the next AI tool, if that ever happens again — picks up from an accurate description instead of stale assumptions.

## 6. Session log

- 2026-07-08 — Migrated project from Windsurf to Claude Code. Fixed a stray git repo rooted at the home directory (was tracking unrelated personal files/SSH keys); re-initialized git scoped to the project folder, added `.gitignore`, set up SSH auth, pushed initial commit to `github.com/arunathavale/RagaDetector`. Reviewed the codebase and wrote this plan. No code changes yet — Phase 1 (database consolidation) is the next actual coding step.
- 2026-07-08 — Phase 1 done. Expanded `config.py::RAAGA_DATABASE` from 2 to all 7 ragas (`Bhairav`, `Asavari`, `Jaunpuri`, `Marwa`, `Puriya` added, each with aroha/avroha/vadi/samvadi/pakad/prahar/forbidden_notes/required_notes/characteristic_phrases). Deleted the inline `RAGA_REGISTRY` + `generate_idealized_histogram()` from both `main.py` and `main_22.py`; both now import `RAAGA_DATABASE` from `config.py` (`main.py` narrows it to `{Yaman, Bhupali}`, matching its original 2-raga scope). `main_22.py`'s forbidden-note penalty check now derives semitone indices from `config.SWARA_MAPPING` instead of hardcoded index lists. Smoke-tested both classifiers against the shared database without audio hardware — imports clean, `config.py`'s own `validate_config()` passes, `classify()`/`check_tie()` run without error on synthetic histograms. Not yet run against live mic input. Next: Phase 2 (eval harness) before touching any classifier logic again.
