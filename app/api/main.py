from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import SILENCE_CHECK_INTERVAL_SECONDS
from app.core.confidence import ConfidenceTracker, evaluate_confidence
from app.core.elimination import (
    confirm_by_candidate_name_match,
    eliminate_by_known_interviewer,
    eliminate_by_silence,
    looks_like_automated_participant_name,
)
from app.core.fusion import compute_probability_distribution
from app.core.registry import ParticipantRegistry, merge_into_turns, ConversationalTurn
from app.core.screenshare_context import ScreenShareTracker, apply_screenshare_context
from app.core.transcript_engine import ParticipantTurnHistory, is_known_interviewer, process_turn
from app.models.schemas import (
    DisplayNameChangeEvent,
    MeetingInitRequest,
    MeetingState,
    ParticipantJoinEvent,
    ParticipantLeaveEvent,
    ParticipantState,
    ScreenShareEvent,
    SpeakingActivityEvent,
    TranscriptEntry,
    WebcamStateEvent,
)

app = FastAPI(title="Sherlock Identity Engine")


class EventEnvelope(BaseModel):
    type: str
    data: dict[str, Any]


class MeetingRuntime:
    def __init__(self, request: MeetingInitRequest) -> None:
        self.meeting_id = request.meeting_id
        self.metadata = request.external_metadata
        self.registry = ParticipantRegistry(self.metadata.calendar_invite)
        self.screen_tracker = ScreenShareTracker()
        self.confidence_tracker = ConfidenceTracker()
        self.turn_histories: dict[str, ParticipantTurnHistory] = {}
        self.total_speaking: dict[str, float] = {}
        self.speaking_start: dict[str, Optional[float]] = {}
        self.raw_transcripts: list[TranscriptEntry] = []
        self.processed_turn_keys: set[tuple[str, float]] = set()
        self.participants_who_have_spoken: set[str] = set()
        self.has_conversation = False
        self.prev_turn_speaker_was_interviewer = False
        self.last_time = 0.0
        self.webcam_state: dict[str, bool] = {}
        self.display_name_history: dict[str, list[str]] = {}
        self.last_silence_check_time: float = float("-inf")
        self.meeting_ended = False

    def process_event(self, event: EventEnvelope) -> dict[str, Any]:
        timestamp = float(event.data.get("timestamp", 0.0))
        self.last_time = max(self.last_time, timestamp)

        if event.type == "participant_join":
            data = ParticipantJoinEvent(**event.data)
            entry = self.registry.register_participant(data.participant_id, data.display_name, timestamp, data.email)
            self.display_name_history.setdefault(data.participant_id, []).append(data.display_name)
            eliminate_by_known_interviewer(entry, data.display_name, data.email, self.metadata)
            confirm_by_candidate_name_match(entry, data.display_name, self.metadata)

        elif event.type == "participant_leave":
            data = ParticipantLeaveEvent(**event.data)
            self.registry.mark_left(data.participant_id, timestamp)

        elif event.type == "webcam_state":
            data = WebcamStateEvent(**event.data)
            self.webcam_state[data.participant_id] = data.webcam_on

        elif event.type == "display_name_change":
            data = DisplayNameChangeEvent(**event.data)
            self.registry.update_display_name(data.participant_id, data.new_display_name)
            self.display_name_history.setdefault(data.participant_id, []).append(data.new_display_name)
            entry = self.registry.get(data.participant_id)
            if entry:
                eliminate_by_known_interviewer(entry, data.new_display_name, entry.email, self.metadata)
                confirm_by_candidate_name_match(entry, data.new_display_name, self.metadata)

        elif event.type == "speaking_activity":
            data = SpeakingActivityEvent(**event.data)
            pid = data.participant_id
            if data.action == "started":
                self.speaking_start[pid] = timestamp
            else:
                start = self.speaking_start.get(pid)
                if start is not None:
                    duration = timestamp - start
                    self.total_speaking[pid] = self.total_speaking.get(pid, 0.0) + duration
                    self.speaking_start[pid] = None
                    if duration > 0.5:
                        self.participants_who_have_spoken.add(pid)
                        if len(self.participants_who_have_spoken) >= 2:
                            self.has_conversation = True

        elif event.type == "screen_share":
            data = ScreenShareEvent(**event.data)
            self.screen_tracker.record_event(data.participant_id, data.action, timestamp)

        elif event.type == "transcript_entry":
            data = TranscriptEntry(**event.data)
            self.raw_transcripts.append(data)
            self._process_new_turns(current_time=timestamp)

        self._run_silence_check(timestamp)
        return self.snapshot()

    def _process_new_turns(self, current_time: float) -> None:
        turns = merge_into_turns(self.raw_transcripts)
        newly_completed: list[ConversationalTurn] = []
        for turn in turns:
            key = (turn.participant_id, turn.started_at)
            if key in self.processed_turn_keys:
                continue
            if turn.ended_at <= current_time:
                newly_completed.append(turn)
                self.processed_turn_keys.add(key)

        if newly_completed:
            self._score_completed_turns(newly_completed)
            last_turn = newly_completed[-1]
            last_entry = self.registry.get(last_turn.participant_id)
            self.prev_turn_speaker_was_interviewer = is_known_interviewer(last_entry) if last_entry else False

    def finalize_pending_turns(self) -> None:
        turns = merge_into_turns(self.raw_transcripts)
        pending = [t for t in turns if (t.participant_id, t.started_at) not in self.processed_turn_keys]
        for turn in pending:
            self.processed_turn_keys.add((turn.participant_id, turn.started_at))
        if pending:
            self._score_completed_turns(pending)

    def _score_completed_turns(self, turns: list[ConversationalTurn]) -> None:
        for turn in turns:
            entry = self.registry.get(turn.participant_id)
            if not entry:
                continue
            history = self.turn_histories.setdefault(turn.participant_id, ParticipantTurnHistory())
            classification = process_turn(
                turn,
                [],
                entry,
                self.metadata,
                history,
                previous_turn_speaker_is_known_interviewer=self.prev_turn_speaker_was_interviewer,
            )
            if classification and entry.evidence_log:
                last_delta = entry.evidence_log[-1].get("applied_delta", 0.0)
                apply_screenshare_context(entry, turn, classification, last_delta, self.screen_tracker)

    def _run_silence_check(self, current_time: float) -> None:
        if current_time - self.last_silence_check_time < SILENCE_CHECK_INTERVAL_SECONDS:
            return
        self.last_silence_check_time = current_time

        active = self.registry.active_entries(current_time)
        for entry in active:
            others_speaking = sum(
                self.total_speaking.get(e.participant_id, 0.0) for e in active
            ) - self.total_speaking.get(entry.participant_id, 0.0)
            sharing = self.screen_tracker.is_sharing_at(entry.participant_id, current_time)
            auto_keyword = looks_like_automated_participant_name(entry.display_name)
            eliminate_by_silence(
                entry,
                current_time,
                participant_total_speaking_seconds=self.total_speaking.get(entry.participant_id, 0.0),
                other_participants_speaking_seconds=others_speaking,
                has_conversational_exchange_occurred=self.has_conversation,
                is_screen_sharing_active=sharing,
                automated_name_keyword=auto_keyword,
            )

    def end_meeting(self) -> dict[str, Any]:
        self.finalize_pending_turns()
        self.meeting_ended = True
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        entries = self.registry.active_entries(self.last_time)
        dist = compute_probability_distribution(entries)
        result = evaluate_confidence(
            entries,
            self.confidence_tracker,
            self.last_time,
            self.has_conversation,
            self.meeting_ended,
        )

        participant_states: dict[str, ParticipantState] = {}
        for entry in self.registry.all_entries():
            participant_states[entry.participant_id] = ParticipantState(
                participant_id=entry.participant_id,
                display_name=entry.display_name,
                display_name_history=self.display_name_history.get(entry.participant_id, [entry.display_name]),
                joined_at=entry.joined_at,
                left_at=entry.left_at,
                webcam_on=self.webcam_state.get(entry.participant_id, False),
                screen_sharing=self.screen_tracker.is_sharing_at(entry.participant_id, self.last_time),
                total_speaking_seconds=self.total_speaking.get(entry.participant_id, 0.0),
                is_speaking=self.speaking_start.get(entry.participant_id) is not None,
                speaking_started_at=self.speaking_start.get(entry.participant_id),
                transcript_entry_count=sum(
                    1 for t in self.raw_transcripts if t.participant_id == entry.participant_id
                ),
            )

        state = MeetingState(
            meeting_id=self.meeting_id,
            external_metadata=self.metadata,
            participants=participant_states,
            transcript=self.raw_transcripts,
        )

        return {
            "meeting_id": self.meeting_id,
            "state": state,
            "confidence": {
                "status": result.status.value,
                "leading_participant_id": result.leading_participant_id,
                "leading_probability": result.leading_probability,
                "leading_log_odds": result.leading_log_odds,
                "explanation": result.explanation,
            },
            "probabilities": dist,
        }


meetings: dict[str, MeetingRuntime] = {}


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/meetings")
def create_meeting(request: MeetingInitRequest) -> dict[str, Any]:
    runtime = MeetingRuntime(request)
    meetings[request.meeting_id] = runtime
    return runtime.snapshot()


@app.post("/meetings/{meeting_id}/events")
def ingest_event(meeting_id: str, event: EventEnvelope) -> dict[str, Any]:
    runtime = meetings.get(meeting_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return runtime.process_event(event)


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str) -> dict[str, Any]:
    runtime = meetings.get(meeting_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return runtime.snapshot()


@app.post("/meetings/{meeting_id}/end")
def end_meeting(meeting_id: str) -> dict[str, Any]:
    runtime = meetings.get(meeting_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="meeting not found")
    return runtime.end_meeting()
