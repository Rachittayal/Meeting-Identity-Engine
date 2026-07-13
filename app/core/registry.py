import itertools
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import (
    LATE_JOIN_PENALTY_LOG_ODDS,
    LATE_JOIN_THRESHOLD_SECONDS,
    LEAVE_GRACE_WINDOW_SECONDS,
    MAX_GAP_FOR_TURN_MERGE_SECONDS,
)
from app.models.schemas import CalendarInvite, TranscriptEntry


class EventSequencer:

    def __init__(self) -> None:
        self.counter = itertools.count(start=1)

    def next_id(self) -> int:
        return next(self.counter)


@dataclass
class ConversationalTurn:
    turn_id: str
    participant_id: str
    text: str
    started_at: float
    ended_at: float
    source_entry_count: int
    min_confidence: Optional[float] = None


def entry_end(entry: TranscriptEntry) -> float:
    return entry.timestamp + (entry.duration_seconds or 0.0)


def merge_into_turns(entries: list[TranscriptEntry],max_gap_seconds: float = MAX_GAP_FOR_TURN_MERGE_SECONDS) -> list[ConversationalTurn]:
    
    if not entries:
        return []

    ordered = sorted(entries, key=lambda e: e.timestamp)

    turns: list[ConversationalTurn] = []
    current_participant: Optional[str] = None
    current_texts: list[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_count = 0
    current_min_confidence: Optional[float] = None

    def flush() -> None:
        if current_participant is None:
            return
        turns.append(
            ConversationalTurn(
                turn_id=str(uuid.uuid4()),
                participant_id=current_participant,
                text=" ".join(current_texts).strip(),
                started_at=current_start,
                ended_at=current_end,
                source_entry_count=current_count,
                min_confidence=current_min_confidence,
            )
        )

    for entry in ordered:
        is_same_speaker = entry.participant_id == current_participant
        gap_seconds = (entry.timestamp - current_end) if current_end is not None else None
        is_within_gap = gap_seconds is not None and gap_seconds <= max_gap_seconds

        if is_same_speaker and is_within_gap:
            current_texts.append(entry.text)
            current_end = max(current_end, entry_end(entry))
            current_count += 1
            if entry.confidence is not None:
                current_min_confidence = (
                    entry.confidence
                    if current_min_confidence is None
                    else min(current_min_confidence, entry.confidence)
                )
            continue

        flush()
        current_participant = entry.participant_id
        current_texts = [entry.text]
        current_start = entry.timestamp
        current_end = entry_end(entry)
        current_count = 1
        current_min_confidence = entry.confidence

    flush()
    return turns


@dataclass
class LedgerEntry:

    participant_id: str
    display_name: str
    email: Optional[str]
    joined_at: float
    left_at: Optional[float] = None
    is_present: bool = True
    log_odds_score: float = 0.0
    evidence_log: list[dict] = field(default_factory=list)
    category_contributions: dict[str, float] = field(default_factory=dict)


class ParticipantRegistry:
    def __init__(self, calendar_invite: Optional[CalendarInvite] = None) -> None:
        self.scheduled_start: Optional[float] = (
            calendar_invite.scheduled_start if calendar_invite else None
        )
        self.entries: dict[str, LedgerEntry] = {}

    def register_participant(self,participant_id: str,display_name: str,joined_at: float,email: Optional[str] = None,) -> LedgerEntry:
        existing = self.entries.get(participant_id)
        if existing is not None:

            existing.is_present = True
            existing.left_at = None
            return existing

        entry = LedgerEntry(
            participant_id=participant_id,
            display_name=display_name,
            email=email,
            joined_at=joined_at,
        )

        if self.is_late_join(joined_at):
            seconds_late = joined_at - self.scheduled_start
            self.apply_registration_delta(
                entry,
                category="join_timing",
                delta=LATE_JOIN_PENALTY_LOG_ODDS,
                reason=(
                    f"Joined {seconds_late:.0f}s after scheduled start; weakly "
                    "correlated with being an unscheduled interviewer rather "
                    "than the invited candidate."
                ),
            )

        self.entries[participant_id] = entry
        return entry

    def mark_left(self, participant_id: str, left_at: float) -> None:
        entry = self.entries.get(participant_id)
        if entry is None:
            return
        entry.is_present = False
        entry.left_at = left_at

    def update_display_name(self, participant_id: str, new_display_name: str) -> None:
        entry = self.entries.get(participant_id)
        if entry is not None:
            entry.display_name = new_display_name

    def get(self, participant_id: str) -> Optional[LedgerEntry]:
        return self.entries.get(participant_id)

    def active_entries(self, current_time: float) -> list[LedgerEntry]:
        return [
            entry
            for entry in self.entries.values()
            if entry.is_present or not self.grace_window_expired(entry, current_time)
        ]

    def all_entries(self) -> list[LedgerEntry]:
        return list(self.entries.values())

    def is_late_join(self, joined_at: float) -> bool:
        if self.scheduled_start is None:
            return False
        return (joined_at - self.scheduled_start) > LATE_JOIN_THRESHOLD_SECONDS

    def grace_window_expired(self, entry: LedgerEntry, current_time: float) -> bool:
        if entry.left_at is None:
            return False
        return (current_time - entry.left_at) > LEAVE_GRACE_WINDOW_SECONDS

    @staticmethod
    def apply_registration_delta(entry: LedgerEntry, category: str, delta: float, reason: str) -> None:
        entry.log_odds_score = entry.log_odds_score + delta
        entry.category_contributions[category] = entry.category_contributions.get(category, 0.0) + delta
        entry.evidence_log.append(
            {
                "category": category,
                "delta": delta,
                "reason": reason,
                "running_score": entry.log_odds_score,
            }
        )