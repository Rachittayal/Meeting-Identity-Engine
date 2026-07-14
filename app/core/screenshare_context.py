from dataclasses import dataclass
from typing import Optional

from app.core.config import SCREENSHARE_AMPLIFICATION_FACTOR
from app.core.ledger import apply_scored_delta
from app.core.registry import ConversationalTurn, LedgerEntry
from app.core.transcript_engine import (
    CANDIDATE_ROLE_CATEGORY_CAP,
    INTERVIEWER_ROLE_CATEGORY_CAP,
    TurnClassification,
    TurnRoleLabel,
)
from app.models.schemas import ScreenShareAction


@dataclass
class ScreenShareInterval:
    started_at: float
    ended_at: Optional[float] = None 


class ScreenShareTracker:

    def __init__(self) -> None:
        self._intervals: dict[str, list[ScreenShareInterval]] = {}

    def record_event(self, participant_id: str, action: ScreenShareAction, timestamp: float) -> None:
        intervals = self._intervals.setdefault(participant_id, [])
        if action == ScreenShareAction.STARTED:
            intervals.append(ScreenShareInterval(started_at=timestamp))
            return

        for interval in reversed(intervals):
            if interval.ended_at is None:
                interval.ended_at = timestamp
                return

    def is_sharing_at(self, participant_id: str, timestamp: float) -> bool:
        for interval in self._intervals.get(participant_id, []):
            end = interval.ended_at if interval.ended_at is not None else float("inf")
            if interval.started_at <= timestamp <= end:
                return True
        return False

    def overlaps_window(self, participant_id: str, window_start: float, window_end: float) -> bool:
        
        for interval in self._intervals.get(participant_id, []):
            end = interval.ended_at if interval.ended_at is not None else float("inf")
            if interval.started_at <= window_end and end >= window_start:
                return True
        return False

_AMPLIFIABLE_LABELS = {TurnRoleLabel.EVALUATIVE_QUESTION, TurnRoleLabel.CANDIDATE_NARRATIVE}


def apply_screenshare_context(
    entry: LedgerEntry,
    turn: ConversationalTurn,
    classification: Optional[TurnClassification],
    step3_applied_delta: float,
    tracker: ScreenShareTracker,
    event_sequence_id: Optional[int] = None,
) -> float:
    
    if classification is None or classification.label not in _AMPLIFIABLE_LABELS:
        return 0.0
    if step3_applied_delta == 0.0:
        return 0.0 

    if not tracker.overlaps_window(turn.participant_id, turn.started_at, turn.ended_at):
        return 0.0  

    amplification = step3_applied_delta * SCREENSHARE_AMPLIFICATION_FACTOR
    category_cap = (
        INTERVIEWER_ROLE_CATEGORY_CAP
        if classification.label == TurnRoleLabel.EVALUATIVE_QUESTION
        else CANDIDATE_ROLE_CATEGORY_CAP
    )

    applied = apply_scored_delta(
        entry,
        category="transcript_role",
        delta=amplification,
        reason=(
            f"Screen-share was active during this turn's window "
            f"[{turn.started_at:.1f}-{turn.ended_at:.1f}], correlating with the "
            f"'{classification.label.value}' classification — amplifying Step 3's evidence "
            f"by {SCREENSHARE_AMPLIFICATION_FACTOR:.0%}, not originating new evidence."
        ),
        event_sequence_id=event_sequence_id,
        category_cap=category_cap,
    )
    return applied