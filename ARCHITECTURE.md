# Architecture Overview

This document serves as a critical, living reference for understanding **Sherlock**'s architecture — the weak-signal fusion engine that identifies the interview candidate on a live call, continuously updates its confidence, and explains every decision it makes. It is intended to get an engineer, reviewer, or evaluator from zero context to a working understanding of the system in a single read. Update this document as the codebase evolves.

## 1. Project Structure

The codebase is organized around a single-responsibility pipeline: each pipeline stage is its own module in `app/core/`, backed by a dedicated test file, with shared contracts isolated in `app/models/`.

```
sherlock/
├── README.md
├── ARCHITECTURE.md            # This document
├── pyproject.toml             # uv-managed dependencies
├── uv.lock
├── docs/
│   └── DATA_CONTRACT.md       # Every brief data field mapped 1:1 to a schema field
├── app/
│   ├── api/
│   │   └── main.py            # FastAPI entrypoint — REST + WebSocket layer
│   ├── models/
│   │   └── schemas.py         # Pydantic contracts for all 7 input data categories
│   └── core/
│       ├── config.py          # Every threshold, env-overridable — nothing hardcoded
│       ├── registry.py        # Step 0: turn merging, re-entrant participant registry
│       ├── ledger.py          # Shared write primitive every signal writes through
│       ├── elimination.py     # Steps 1–2: structural elimination + silence monitoring
│       ├── transcript_engine.py   # Step 3: LLM-scoped transcript role classification
│       ├── screenshare_context.py # Step 4: proportional amplifier (not independent)
│       ├── fusion.py          # Step 5: capped log-odds ledger + cross-participant softmax
│       └── confidence.py      # Step 6: four-quadrant status, hysteresis, entropy
└── tests/                     # 101 tests, one file per module above
```

**Design rationale:** every architectural boundary in this tree maps directly onto a stage of the fusion pipeline (Section 2). There is no code that doesn't belong to a named pipeline step, which keeps the "where do I add a new signal?" question trivial to answer — you add a module to `app/core/`, wire it to the ledger, and add its test file.

## 2. High-Level System Diagram

```
┌───────────────────────────────────────────────────────────────┐
│  Meeting Platform Ingestion (Zoom / Meet / Teams)                │
│  Participant events · audio/video stream refs · transcript       │
│  · external metadata (calendar, candidate name, interviewers)    │
└─────────────────────────────┬─────────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 0 — Normalization                                          │
│  Turn merging (prevents diarization double-counting)              │
│  Re-entrant participant registry (late-join penalty, grace-window │
│  leave/rejoin handling)                                            │
└─────────────────────────────┬─────────────────────────────────┘
                              ▼
        ┌──────────────────────┼──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼                      ▼
┌───────────────────┐ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ Steps 1–2            │ │ Step 3               │ │ Step 4               │ │ Every signal above   │
│ Structural            │ │ Transcript role      │ │ Screen-share         │ │ writes to one         │
│ elimination +          │ │ classification        │ │ context               │ │ shared, per-           │
│ silence monitoring     │ │ (LLM-scoped)          │ │ (amplifier, not       │ │ participant ledger     │
│                       │ │                       │ │  independent)         │ │                        │
└───────────────────┘ └────────────────────┘ └────────────────────┘ └────────────────────┘
        │                      │                      │                      │
        └──────────────────────┴──────────────────────┴──────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 5 — Fusion                                                  │
│  Log-odds ledger · per-signal-category caps · cross-participant   │
│  softmax normalization (computed once, at read time)               │
└─────────────────────────────┬─────────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 6 — Confidence Evaluation                                   │
│  Four-quadrant status (magnitude × entropy) · hysteresis ·         │
│  temporarily-absent state · terminal unresolved output ·           │
│  metadata-mismatch detection                                       │
└─────────────────────────────┬─────────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  Step 7 — API Layer                                               │
│  REST events ingestion + WebSocket push of live state              │
└─────────────────────────────┬─────────────────────────────────┘
                              ▼
       {candidate_id, status, confidence, full evidence trail}
                              ▼
        Sherlock's fraud detectors + live confidence dashboard
```

Every box above corresponds to a real, tested module in `app/core/` — this is a diagram of how the code is actually organized, not an aspirational sketch.

## 3. Core Components

### 3.1. Frontend / Consumers

**Name:** Sherlock fraud-detection stack + live confidence dashboard

**Description:** Sherlock is not a standalone UI product — it is an identity-resolution layer consumed by downstream fraud detectors (deepfake detection, voice cloning, behavioral analysis) that need to know *which* participant stream to analyze, plus a live dashboard that surfaces confidence and evidence trail to a human reviewer in real time.

**Technologies:** WebSocket push for live state, REST for event ingestion.

**Deployment:** Consumed as an internal service by other Sherlock components; not independently deployed as a user-facing app in this prototype.

### 3.2. Backend Services

#### 3.2.1. Identity Resolution Engine (this project)

**Name:** Sherlock Meeting-Identity-Engine

**Description:** A weak-signal fusion engine that ingests meeting-platform events (participant joins/leaves, transcript turns, screen-share activity, calendar metadata) and continuously outputs the most likely candidate identity, a confidence status, and a full, human-readable evidence trail. Built as inference under uncertainty rather than lookup, because no single signal (name matching, "first to speak," "camera on") survives real-world interview conditions.

**Technologies:** Python 3.12+, FastAPI (REST + WebSocket), Pydantic (schema contracts), uv (dependency management), pytest (101 tests).

**Deployment:** `uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000`; platform-agnostic by design, so it can sit behind a Zoom/Meet/Teams bot or a scripted test harness with no code changes.

#### 3.2.2. Turn Classifier (pluggable sub-component of Step 3)

**Name:** `TurnClassifier`

**Description:** A bounded, per-turn semantic classifier that labels each conversational turn as `evaluative_question | informational_question | candidate_narrative | self_introduction | none`, with a confidence score. Ships by default as a regex-based heuristic classifier (zero external API dependency, safe fallback to `none` on unfamiliar phrasing), and is designed to be swapped for a real LLM with no changes to any downstream logic.

**Technologies:** Regex-based heuristics by default; interface designed for drop-in LLM substitution.

**Deployment:** In-process module (`transcript_engine.py`); no separate service boundary.

## 4. Data Stores

### 4.1. Per-Participant Ledger

**Name:** Shared evidence ledger

**Type:** In-process, per-meeting append-only ledger (not an external database in the current prototype)

**Purpose:** The single write path every signal — structural elimination, silence monitoring, transcript classification, screen-share context — writes through. This is what makes the system's explanation trustworthy: every confidence number traces back to a specific logged delta, not a post-hoc summary.

**Key Structures:** per-participant log-odds accumulator, per-signal-category caps, full evidence/decision trail.

### 4.2. Configuration Store

**Name:** `config.py`

**Type:** Environment-overridable static configuration (no external store)

**Purpose:** Centralizes every threshold used across the pipeline (silence timeout, category caps, hysteresis bounds) so nothing is hardcoded inside a signal module — a deliberate choice to keep calibration a config change, not a code change.

## 5. External Integrations / APIs

**Service:** Meeting platforms — Zoom, Google Meet, Microsoft Teams

**Purpose:** Source of participant join/leave events, transcript turns, screen-share activity, and (via calendar) candidate/interviewer metadata.

**Integration Method:** Platform-agnostic event ingestion — nothing in `app/core/` assumes whether events originate from a real meeting bot or a scripted test scenario, by design.

**Service:** LLM provider (optional, Step 3 only)

**Purpose:** Optional upgrade path for the `TurnClassifier` — richer semantic classification of conversational turns than the default regex heuristic provides.

**Integration Method:** Swappable interface (`TurnClassifier` type); no other module depends on which implementation is active.

## 6. Deployment & Infrastructure

**Runtime:** Python 3.12+, managed via `uv` (dependencies pinned in `pyproject.toml` / `uv.lock`)

**Service Layer:** FastAPI, exposing REST endpoints for event ingestion and a WebSocket channel for live confidence/state push (Step 7)

**CI/CD Pipeline:** Not yet defined for this prototype stage — see Section 9

**Monitoring & Logging:** The evidence ledger itself functions as the system's structured log — every decision is written with enough detail (named function, source quote, delta) to support offline threshold tuning and audit, without a separate logging/observability stack layered on top

## 7. Security Considerations

**Authentication:** Not yet implemented — out of scope for the current identity-resolution prototype; would be required at the API layer (Section 3.2.1) before any production deployment behind real meeting platforms

**Authorization:** Not yet implemented

**Data Handling:** No biometric data (photo/voice) is ingested or stored by design (see Assumption 1, Section 10) — identity is inferred from behavioral and metadata signals only, which meaningfully narrows the system's privacy/data-sensitivity surface area relative to a face- or voice-matching approach

**Key Practice:** `participant_id` is treated as the only stable identity anchor; display names are tracked as mutable history and never overwritten, which keeps the audit trail honest even as participants rename themselves mid-call

## 8. Development & Testing Environment

**Local Setup:**
```bash
git clone https://github.com/Rachittayal/Meeting-Identity-Engine.git
cd Meeting-Identity-Engine
uv sync
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

**Running the evaluation script:**
```bash
uv run python scripts/evaluate_meeting_flow.py
```

**Testing Framework:** pytest — 101 tests across 7 files, one per `app/core/` module, written and run immediately after each module rather than retrofitted at the end. Several tests caught and fixed real bugs during development (see Section 9).

**Code Quality:** Pydantic-enforced schema contracts at every module boundary (`app/models/schemas.py`); every threshold externalized to `config.py` rather than hardcoded, which doubles as a form of self-documentation for tunable behavior.

## 9. Design Decisions & Known Limitations

This section replaces a generic "Future Considerations" placeholder with the project's actual, tested findings — stated honestly rather than aspirationally.

### 9.1. Why Elimination-Then-Confirmation, Fused Incrementally

The core design decision: every participant is scored independently, in isolation, by every signal — and only one place in the entire system (Step 5's softmax) ever compares participants against each other.

- **Elimination signals** (Steps 1–2) exploit cheap, static, closed-world data first — display name vs. known interviewer, email vs. calendar attendee — because that is the highest information-gain-per-unit-of-compute move available before anyone has spoken. Silence monitoring adds a purely behavioral fallback that catches bots/observers by *behavior*, not by name, so it can't be spoofed and doesn't depend on a keyword list staying current.
- **Confirmation signals** (Step 3) apply richer, noisier semantic evidence — is a participant asking evaluative questions or answering with first-person narrative? This is the system's only LLM-scoped decision point (Section 3.2.2).
- **Fusion** (Step 5) accumulates every signal's contribution as a running, per-category-capped log-odds sum, so no single evidence type can mathematically force false certainty. Cross-participant softmax comparison happens exactly once, at read time — which is what makes the system naturally robust to participants joining or leaving mid-call, since nothing upstream needs to change.
- **Confidence evaluation** (Step 6) decides "who" and "how sure" together, looking at both the magnitude of the leading candidate's evidence and the entropy of the whole distribution — because a confident two-way tie and genuine early-stage uncertainty look identical under a bare probability margin, but mean very different things to a human reviewer.

### 9.2. Where the LLM Fits — And Deliberately Doesn't

The LLM (when enabled) is used **only** as a bounded, per-turn classifier inside Step 3 — never as the component that decides who the candidate is, and never as the component that writes the final explanation. Fusion, confidence scoring, and the human-facing explanation are all deterministic arithmetic over a logged evidence trail. This was a deliberate trade-off: an LLM asked to summarize *why* it reached a conclusion can produce a plausible-sounding narrative that silently diverges from what actually happened. Given the brief's explicit weighting toward explainability, that structural guarantee mattered more than a more "impressive"-sounding fully-LLM pipeline.

### 9.3. Assumptions

1. **No biometric reference exists.** Webcam/audio stream references are given, never a verified photo or voice sample — face recognition and voiceprint matching are out of scope by data contract, not by choice.
2. **External metadata is a prior, never ground truth.** Calendar invites and candidate names are human-entered and can be wrong; every signal derived from this category is capped, correctable evidence, never a hard disqualifier.
3. **`participant_id` is the only stable identity anchor.** Display names are mutable and tracked as history, never overwritten.
4. **No keyword-list-based bot detection.** An earlier design used a list of known notetaking-tool names as a detection signal; it was deliberately removed, since such a list is permanently incomplete and carries irreducible false-positive risk against real participant names. Bot/observer detection relies entirely on behavioral silence monitoring instead.
5. **Timestamps are seconds-elapsed-since-meeting-start**, so synthetic test scenarios and real meeting data are directly comparable.
6. **Ingestion is platform-agnostic by design.** Nothing in `app/core/` assumes whether events originate from a real Zoom/Meet/Teams bot or a scripted test scenario.

### 9.4. Roadmap — Audio & Video as a Level 2

The current prototype identifies the candidate using structural metadata and transcript semantics alone — a deliberate, defensible scope that requires no infrastructure beyond what the brief guarantees, and already handles every named failure case (nicknames, wrong names, multiple interviewers, mid-call renaming, silent observers). A production version would add two further, independent evidence layers on top of this foundation, not replace it:

- **Audio, as a behavioral signal layer:** speaker diarization timing, turn-taking cadence, and speaking-duration patterns as direct numerical signals — how participation is distributed across the call, not just what was said.
- **Video, as a presence and engagement signal layer:** camera-on consistency and presence patterns as an additional behavioral fingerprint, plus optional face verification where a reference image is available, as a genuinely independent confirming signal.

Architecturally, this matters because transcript, audio, and video are three genuinely independent evidence sources that fail in different, uncorrelated ways — exactly the property the fusion design is built around. Adding them means two new signal modules feeding the *same* ledger and confidence evaluator that already exist; the architecture doesn't need to change to accommodate this, only grow.

### 9.5. Known Limitations (stated honestly)

- **No biometric cross-device linkage.** The same person joining from two devices (laptop + phone) currently appears as two separate participants — nothing in the given data schema exposes a signal to link them.
- **Interpreters and proxies are out of scope.** Someone speaking on another person's behalf looks structurally identical to genuine first-person narrative.
- **Confidence scores are directionally meaningful, not statistically calibrated.** Without a labeled dataset of real interviews, the log-odds deltas are reasoned, hand-tuned weights — internally consistent for ranking and abstention decisions, but not rigorously calibrated real-world probabilities. Every decision is logged with enough detail to support future offline calibration.
- **The default heuristic classifier is regex-based, not LLM-backed.** Genuinely novel phrasing outside its pattern list is missed — safely, by falling through to `none` and contributing no evidence rather than misclassifying.
- **Accuracy is not yet measured against a labeled dataset.** The scenario test suite (labeled fixtures from the brief's six named cases) is the next piece of work; the current 101 tests establish unit-level correctness, not an end-to-end accuracy number.

## 10. Evaluation Summary

Every module was tested immediately after being built, not retrofitted at the end — 101 unit tests across 7 test files, one per `app/core/` module. Edge cases explicitly covered beyond the six named in the original brief:

- Name collision between the candidate and an interviewer sharing a first name (elimination correctly suppressed rather than firing)
- A wrong or mismatched calendar candidate name (system degrades gracefully rather than crashing or falsely eliminating)
- A single-participant meeting with strongly negative evidence (correctly reports `UNCERTAIN`, not a false `CONFIRMED` — a real bug caught and fixed during development)
- A category cap saturating and then needing to accept a corrective delta in the opposite direction (another real bug caught and fixed — the original cap logic silently blocked corrections after saturation)
- A confirmed candidate briefly disconnecting (`TEMPORARILY_ABSENT`, not a false promotion of someone else)
- A meeting ending without ever reaching confident identification (`UNRESOLVED`, not a silent low-confidence guess presented as fact)
- Ten-plus minutes of active conversation with no participant crossing any evidence threshold (`METADATA_MISMATCH` — flags that the calendar data itself may be wrong)

## 11. Project Identification

**Project Name:** Sherlock — Real-Time Interview Candidate Identification (Meeting-Identity-Engine)

**Repository URL:** https://github.com/Rachittayal/Meeting-Identity-Engine

**Primary Contact/Team:** Rachit Tayal

**Date of Last Update:** 2026-07-14

## 12. Glossary / Acronyms

**Elimination signal:** A cheap, static, closed-world signal (e.g., name matching a known interviewer) used to rule participants *out* early, before behavioral data is available.

**Confirmation signal:** A richer, noisier semantic signal (e.g., transcript role classification) used to build positive evidence *toward* a candidate.

**Log-odds ledger:** The running, per-participant, per-signal-category-capped accumulator that every signal writes its contribution to; the basis for both the final confidence score and the human-readable explanation.

**Softmax normalization:** The single point (Step 5, at read time) where participants are compared against each other, rather than scored in isolation — what allows the system to naturally handle participants joining or leaving mid-call.

**Four-quadrant status:** Step 6's confidence classification, which considers both the magnitude of the leading participant's evidence and the entropy of the whole distribution, rather than a single probability margin.

**Hysteresis:** The mechanism preventing a confidence status from flapping back and forth across a threshold on marginal, noisy evidence.

**`TurnClassifier`:** The pluggable interface for Step 3's per-turn semantic classifier; ships as a regex heuristic by default, swappable for an LLM with no downstream changes.
