# Sherlock Identity Engine

## Prototype status
This repository contains a working prototype for the Sherlock internship challenge. It processes a meeting scenario event stream and produces a real-time candidate ranking with evidence-backed explanations.

The current implementation is functional for the provided sample scenario, but it is not yet a production-grade solution for live audio/video meetings.

## What the pipeline does
The prototype combines several weak signals to estimate which participant is most likely to be the interview candidate:

- participant join and leave timing
- display-name matching against the candidate name and known interviewer names
- email-based elimination against interviewer identities from calendar metadata
- transcript-role classification for evaluative questions and candidate narratives
- turn-taking patterns after interviewer turns
- screen-share context amplification
- confidence evaluation with explanation text

## Main modules
- [main.py](main.py) – orchestrates the event loop and prints the evolving candidate state
- [app/core/transcript_engine.py](app/core/transcript_engine.py) – classifies transcript turns into interview-relevant roles
- [app/core/elimination.py](app/core/elimination.py) – removes unlikely participants using name, email, and silence heuristics
- [app/core/screenshare_context.py](app/core/screenshare_context.py) – amplifies evidence when a participant shares their screen during relevant turns
- [app/core/confidence.py](app/core/confidence.py) – turns the evidence ledger into a confidence status and explanation
- [app/core/registry.py](app/core/registry.py) – stores participant state and the evidence ledger

## How to run
From the project root:

```bash
.venv\Scripts\python.exe main.py
```

## Verified behavior
I verified the current implementation by running the scenario through the meeting runtime. The engine produced a ranked participant list, reached a confirmed candidate state for the supplied scenario, and emitted an explanation that remained stable through the final grace-window state.

## Current limitations
This is still a heuristic prototype rather than a full production system. Important gaps remain:

- it consumes a pre-recorded event scenario rather than live meeting streams
- it does not ingest actual audio or video data
- it uses rule-based transcript heuristics rather than richer ML or LLM-based reasoning
- it does not yet expose a web API or streaming service interface
- it has an initial automated regression test for confidence evaluation, but the broader suite is still limited

## Recommended next steps
To bring this closer to the challenge brief, the next logical improvements are:

1. add a real-time ingestion layer for live meeting events
2. integrate audio/video-based candidate evidence
3. add a richer classifier or LLM-backed reasoning stage
4. add automated tests and a simple API wrapper
5. produce a demo video and architecture diagram for submission
