# Sherlock — Real-Time Interview Candidate Identification

**A weak-signal fusion engine that identifies the interview candidate on a live call, continuously updates its confidence, and explains every decision it makes.**

Built for the Sherlock Internship Challenge — designing the identity-resolution layer that lets Sherlock's fraud detectors (deepfake detection, voice cloning, behavioral analysis) know *which* participant's stream to actually analyze.

---

## Table of Contents

- [The Problem, In One Paragraph](#the-problem-in-one-paragraph)
- [Why This Isn't a Lookup Problem](#why-this-isnt-a-lookup-problem)
- [Architecture](#architecture)
- [The Core Idea: Elimination + Confirmation, Fused Incrementally](#the-core-idea-elimination--confirmation-fused-incrementally)
- [Where the LLM Fits — And Where It Deliberately Doesn't](#where-the-llm-fits--and-where-it-deliberately-doesnt)
- [Project Structure](#project-structure)
- [Setup & Running the Tests](#setup--running-the-tests)
- [Walking Through a Real Scenario](#walking-through-a-real-scenario)
- [Assumptions](#assumptions)
- [Current Status & What's Left](#current-status--whats-left)
- [Roadmap: Audio & Video as Level 2](#roadmap-audio--video-as-level-2)
- [Evaluation](#evaluation)

---

## The Problem, In One Paragraph

Sherlock's fraud detectors need to know, at every moment of a live interview, which video/audio stream belongs to the *candidate* — not the interviewer, not a silent observer, not a notetaking bot. That sounds trivial until you look at how real interviews actually go: candidates join as **"MacBook Pro,"** under nicknames, sometimes with a recruiter-entered name that's flat wrong. Multiple interviewers sit in. Silent observers lurk. People rename themselves mid-call. No single rule — name matching, "first to speak," "camera on" — survives contact with all of these at once. This project treats identification as **inference under uncertainty**, not lookup.

## Why This Isn't a Lookup Problem

Every naive approach breaks on a named failure case from the brief:

| Naive approach | Breaks on |
|---|---|
| Match display name to calendar | Nicknames, generic device names, wrong data entry |
| "First person to speak is the interviewer" | Multiple interviewers, candidates who ask questions too |
| "Person with camera on is the candidate" | Cultural/technical variance, silent observers with cameras on |
| Trust the calendar invite | Recruiters mistype names and emails constantly |

None of these signals is individually trustworthy — but each one is **wrong in a different, uncorrelated way**. That's the whole design principle this system is built on: combine several independently-failing weak signals so no single failure mode can dominate the outcome.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Meeting Platform Ingestion (Zoom / Meet / Teams)          │
│  Participant events · audio/video stream refs · transcript │
│  · external metadata (calendar, candidate name, interviewers)│
└───────────────────────┬─────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 0 — Normalization                                     │
│  Turn merging (prevents diarization double-counting)         │
│  Re-entrant participant registry (late-join penalty,         │
│  grace-window leave/rejoin handling)                         │
└───────────────────────┬─────────────────────────────────┘
                         ▼
        ┌────────────────┼────────────────┬──────────────────┐
        ▼                ▼                ▼                  ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Steps 1–2       │ │ Step 3        │ │ Step 4        │ │ every signal        │
│ Structural       │ │ Transcript     │ │ Screen-share  │ │ above writes to     │
│ elimination +     │ │ role           │ │ context       │ │ one shared,         │
│ silence           │ │ classification │ │ (amplifier,   │ │ per-participant      │
│ monitoring        │ │ (LLM-scoped)   │ │  not          │ │ ledger               │
│                   │ │                │ │  independent) │ │                      │
└───────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘
        │                │                │                  │
        └────────────────┴────────────────┴──────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5 — Fusion                                             │
│  Log-odds ledger, per-signal-category caps, cross-            │
│  participant softmax normalization                           │
└───────────────────────┬─────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6 — Confidence Evaluation                               │
│  Four-quadrant status (magnitude × entropy) · hysteresis ·    │
│  temporarily-absent state · terminal unresolved output ·      │
│  metadata-mismatch detection                                  │
└───────────────────────┬─────────────────────────────────┘
                         ▼
     {candidate_id, status, confidence, full evidence trail}
                         ▼
      Sherlock's fraud detectors + live confidence dashboard
```

Every box above is a real, tested module in `app/core/` — this isn't a conceptual diagram describing something aspirational; it's how the code is actually organized.

---

## The Core Idea: Elimination + Confirmation, Fused Incrementally

The single most important design decision in this project: **every participant is scored independently, in isolation, by every signal — and only one place in the whole system ever compares participants against each other.**

**Elimination signals** (Steps 1–2) exploit cheap, static, closed-world data first — a display name matching a known interviewer, or an email matching a calendar attendee — because that's the highest information-gain-per-unit-of-compute move available before anyone has even spoken. Silence monitoring adds a purely behavioral fallback: an obvious bot or silent observer gets caught by *behavior*, not by what they're named, which means it can't be spoofed by a display name and doesn't depend on any keyword list staying current.

**Confirmation signals** (Step 3) apply richer but noisier semantic evidence — is this participant asking evaluative questions ("walk me through your approach to...") or answering with first-person narrative about their own experience? This is where the system's only LLM-scoped decision point lives (see below).

**Fusion** (Step 5) accumulates every signal's contribution as a running **log-odds sum**, capped per signal category so no single evidence type can mathematically force false certainty. Cross-participant comparison — softmax normalization — happens exactly once, at read time, which is what makes the system naturally robust to people joining or leaving mid-call: nothing upstream needs to change, the system just re-normalizes over whoever is currently active.

**Confidence evaluation** (Step 6) is where "who" and "how sure" get decided together rather than as two bolted-together systems. It looks at both the *magnitude* of the leading participant's evidence and the *entropy* of the whole distribution — because a confident two-way tie and genuine early-stage uncertainty can look identical if you only ever look at a probability margin, and they mean very different things to a human reviewing the output.

---

## Where the LLM Fits — And Where It Deliberately Doesn't

This was a first-principles decision, not a default: the LLM is used **only** as a bounded, per-turn classifier inside Step 3 — never as the component that decides who the candidate is, never as the component that writes the final explanation.

- **Input:** one conversational turn, plus 1–2 turns of preceding context.
- **Output:** a structured label (`evaluative_question | informational_question | candidate_narrative | self_introduction | none`) plus a confidence score.
- **That's it.** The classifier is fully swappable (`TurnClassifier` type in `transcript_engine.py`) — a regex-based heuristic classifier ships by default so the system runs offline with zero API dependency, and a real LLM can be substituted with no changes to any downstream logic.

Everything downstream — fusion, confidence scoring, and critically, **the explanation shown to a human** — is deterministic arithmetic over a logged evidence trail. This was a deliberate trade-off: an LLM asked to summarize *why* it reached a conclusion can produce a plausible-sounding narrative that silently diverges from what actually happened. Here, the explanation *is* the update log, in plain English — every confidence number traces back to a named function, a specific quote, and a specific delta. Given the brief's explicit weighting toward explainability, that structural guarantee mattered more than a more "impressive"-sounding fully-LLM pipeline.

---

## Project Structure

```
sherlock/
├── README.md
├── pyproject.toml            # uv-managed dependencies
├── uv.lock
├── docs/
│   └── DATA_CONTRACT.md      # every brief data field → schema field, 1:1
├── app/
│   ├── models/
│   │   └── schemas.py        # Pydantic contracts for all 7 data categories
│   └── core/
│       ├── config.py         # every threshold, env-overridable — nothing hardcoded
│       ├── registry.py       # Step 0: turn merging, re-entrant participant registry
│       ├── ledger.py         # shared write primitive every signal writes through
│       ├── elimination.py    # Steps 1–2: structural elimination + silence
│       ├── transcript_engine.py   # Step 3: LLM-scoped role classification
│       ├── screenshare_context.py # Step 4: proportional amplifier
│       ├── fusion.py         # Step 5: cross-participant softmax
│       └── confidence.py     # Step 6: four-quadrant status, hysteresis, entropy
└── tests/                    # 101 tests, one file per module above
```

## Setup & Running the Tests

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync                    # installs dependencies from pyproject.toml / uv.lock
uv run pytest tests/ -v    # 101 tests, all passing, zero external API dependency
```

Every module above has its own dedicated test file — the tests aren't an afterthought bolted on at the end; they were written and run immediately after each module, including tests that specifically caught and fixed real bugs during development (a cap-clamping bug that could silently block corrective evidence, a false-positive risk in bot-name matching, a quadrant-classification edge case for single-participant meetings). See [Evaluation](#evaluation) for specifics.

---

## Walking Through a Real Scenario

Concrete beats abstract, so here's the system's actual behavior on the brief's core failure case:

**t=0s** — Four participants join: `"MacBook Pro"`, `"Sarah K."`, `"Ronnie"`, `"Guest 8823"`. Calendar says candidate is *Ronald Fernandes*, interviewer is *Sarah Kapoor*.
`"Sarah K."` fuzzy-matches the interviewer name → eliminated immediately, zero behavioral data needed. `"Ronnie"` weakly matches *Ronald Fernandes* → small positive nudge. Status: `UNCERTAIN`.

**t=45s** — Sarah asks: *"Ronnie, walk me through your background."* Ronnie responds with 45 seconds of first-person narrative.
Step 3 classifies Sarah's turn as `evaluative_question` (further reinforcing her elimination) and Ronnie's response as `candidate_narrative`. Status climbs toward `EARLY_LEAN`.

**t=3min** — `"MacBook Pro"` still hasn't spoken. Silence monitoring hasn't fired yet (5-minute threshold) — the system correctly stays silent about this rather than guessing early.

**t=8min** — Ronnie has answered 6 more questions; `"MacBook Pro"` has now crossed the silence threshold with zero speaking activity while others actively talk. Status: `CONFIRMED` for Ronnie, with an evidence trail reading (roughly): *name similarity to calendar (weak) → addressed by name and responded (strong) → 6 of 7 subsequent turns classified as candidate narrative (strong, established pattern) → co-participant crossed silence threshold with no corroborating activity (supporting).*

**t=12min** — Ronnie renames to `"Ron F."` mid-call. Because the system tracks by `participant_id`, not display name, this doesn't reset anything — it's folded in as one more small corroborating data point.

No single rule made any of these calls. Every step is a capped, logged contribution to a running score, and the explanation above is generated directly from that log — not written after the fact.

---

## Assumptions

Stated explicitly because they materially shape the design:

1. **No biometric reference exists.** The data contract gives webcam/audio stream *references*, never a verified photo or voice sample of the candidate. Face recognition and voiceprint matching are out of scope **by data contract, not by choice** — identity is inferred from behavior and metadata, never biometrics.
2. **External metadata is a prior, never ground truth.** Calendar invites and candidate names are human-entered and can be wrong. Every signal derived from this category is capped, correctable evidence — never a hard, permanent disqualifier.
3. **`participant_id` is the only stable identity anchor.** Display names are mutable and tracked as history, never overwritten.
4. **No keyword-list-based bot detection.** An earlier design used a list of known notetaking-tool names (`"Otter.ai"`, `"Fireflies"`, etc.) as a detection signal. It was deliberately removed: such a list is permanently incomplete (new tools ship constantly) and, even with careful matching, carries irreducible false-positive risk against real participant names. Bot/observer detection relies entirely on **behavioral** silence monitoring, which can't be spoofed by a display name.
5. **Timestamps are seconds-elapsed-since-meeting-start**, so synthetic test scenarios and real meeting data are directly comparable.
6. **Ingestion is platform-agnostic by design.** Nothing in `app/core/` assumes whether events originate from a real Zoom/Meet/Teams bot or a scripted test scenario.

---

## Current Status & What's Left

| Component | Status |
|---|---|
| Data contract (`schemas.py`) | ✅ Complete |
| Step 0 — Normalization (`registry.py`) | ✅ Complete, 11 tests |
| Steps 1–2 — Elimination + silence (`elimination.py`) | ✅ Complete, 16 tests |
| Ledger (`ledger.py`) | ✅ Complete, 7 tests |
| Step 3 — Transcript classification (`transcript_engine.py`) | ✅ Complete, 21 tests |
| Step 4 — Screen-share context (`screenshare_context.py`) | ✅ Complete, 14 tests |
| Step 5 — Fusion (`fusion.py`) | ✅ Complete, 9 tests |
| Step 6 — Confidence evaluation (`confidence.py`) | ✅ Complete, 23 tests |
| API layer (FastAPI routes, event dispatcher) | ⬜ Not yet built |
| Scenario test suite (six named brief edge cases as fixtures) | ⬜ Not yet built |
| Demo video | ⬜ Not yet recorded |

The core identification pipeline — every signal, the fusion mechanism, and the confidence evaluator — is complete and tested end-to-end at the unit level. What remains is wiring it to a live API surface and building the labeled scenario fixtures that turn "the logic is correct" into "here's a measured accuracy number."

---

## Roadmap: Audio & Video as Level 2

The current prototype identifies the candidate using **structural metadata and transcript semantics** — display names, calendar data, and what participants say. That's a deliberate, defensible scope for this stage: it requires no additional infrastructure beyond what the brief guarantees is available, and it's already enough to handle every named failure case (nicknames, wrong names, multiple interviewers, renaming mid-call, silent observers).

A production version of Sherlock would add two further, independent layers of evidence on top of this foundation, not replace it:

**Audio, as a behavioral signal layer.** With a separate audio stream per participant, the system could move beyond transcript content alone to use speaker diarization timing, turn-taking cadence, and speaking-duration patterns as direct numerical signals — not just *what* was said, but *how* participation is distributed across the call. A candidate typically answers in longer narrative turns and responds consistently after being prompted; an interviewer's speaking pattern looks different in shape, independent of the words used.

**Video, as a presence and engagement signal layer.** With a separate webcam stream per participant, the system could track camera-on consistency and presence patterns as an additional behavioral fingerprint — and, where a reference image is available, optional face verification as a genuinely independent confirming signal, distinct from anything metadata or transcript can provide.

The reason this matters architecturally, not just as a feature list: **transcript tells the system what was said, audio tells it how participation behaved, video tells it who was consistently present** — three genuinely independent evidence sources that fail in different, uncorrelated ways, which is exactly the property this system's entire fusion design is built around. Adding them would mean two new signal modules feeding the *same* ledger and confidence evaluator that already exist — the architecture doesn't need to change to accommodate this, only grow.

In one line: *while the current prototype uses metadata and transcript-based heuristics, a more reliable production system would ingest per-participant audio and video streams to add behavioral evidence — speaking patterns, turn-taking, webcam engagement, and optional face verification — making candidate identification more robust in real time.*

---

## Evaluation

**How the system was tested:** every module was tested immediately after being built, not retrofitted at the end — 101 unit tests across 7 test files, one per `app/core/` module, run continuously throughout development rather than as a final pass.

**Edge cases explicitly covered by tests**, beyond the six named in the brief:
- Name collision between the candidate and an interviewer sharing a first name (elimination correctly suppressed rather than firing)
- A wrong/mismatched calendar candidate name (system degrades gracefully, doesn't crash or falsely eliminate)
- A single-participant meeting where that participant has strongly negative evidence (correctly reports `UNCERTAIN`, not a false `CONFIRMED` — this was an actual bug caught and fixed during development, not a hypothetical)
- A category cap saturating and then needing to accept a corrective delta in the opposite direction (another real bug caught and fixed — the original cap logic silently blocked corrections after saturation)
- A confirmed candidate briefly disconnecting (`TEMPORARILY_ABSENT`, not a false promotion of someone else)
- A meeting ending without ever reaching confident identification (`UNRESOLVED`, not a silent low-confidence guess presented as fact)
- Ten-plus minutes of active conversation with no participant crossing any evidence threshold (`METADATA_MISMATCH` — flags that the calendar data itself may be wrong, rather than continuing to report generic uncertainty)

**Accuracy:** not yet measured against a labeled dataset — the scenario test suite (labeled fixtures built from the brief's six named cases) is the next piece of work, and is what would produce an actual reportable accuracy number rather than unit-level correctness claims.

**Limitations, stated honestly:**

- **No biometric cross-device linkage.** If the same person joins from two devices (laptop + phone, common on unstable connections), the system currently sees two separate participants — nothing in the given data schema exposes a signal to link them.
- **Interpreters and proxies are out of scope.** Someone speaking on another person's behalf would look structurally identical to genuine first-person narrative; nothing in the given data distinguishes this.
- **Confidence scores are directionally meaningful, not statistically calibrated.** Without a labeled dataset of real interviews to calibrate against, the log-odds deltas are reasoned, hand-tuned weights — internally consistent for ranking and abstention decisions, but not rigorously calibrated real-world probabilities. The architecture supports future calibration (every decision is logged with enough detail for offline threshold tuning) but the prototype ships with defaults, not fitted parameters.
- **The default heuristic classifier is regex-based**, not LLM-backed, in its current default configuration — genuinely novel phrasing outside its pattern list will be missed (though safely: it falls through to `NONE` and contributes no evidence, rather than misclassifying).
