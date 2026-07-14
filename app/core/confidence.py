import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.config import (
    CONFIDENCE_ENTROPY_ENTER_THRESHOLD,
    CONFIDENCE_MAGNITUDE_ENTER_THRESHOLD,
    CONFIRMED_ENTROPY_EXIT_THRESHOLD,
    CONFIRMED_MAGNITUDE_EXIT_THRESHOLD,
    METADATA_MISMATCH_MAX_SCORE_FLOOR,
    METADATA_MISMATCH_TIME_THRESHOLD_SECONDS,
)
from app.core.fusion import get_ranked_participants
from app.core.registry import LedgerEntry


class ConfidenceStatus(str, Enum):
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    EARLY_LEAN = "early_lean"
    UNCERTAIN = "uncertain"
    TEMPORARILY_ABSENT = "temporarily_absent"
    METADATA_MISMATCH = "metadata_mismatch"
    UNRESOLVED = "unresolved"


@dataclass
class ConfidenceTracker:

    current_status: ConfidenceStatus = ConfidenceStatus.UNCERTAIN
    confirmed_participant_id: Optional[str] = None


@dataclass
class ConfidenceResult:
    status: ConfidenceStatus
    leading_participant_id: Optional[str]
    leading_probability: float
    leading_log_odds: float
    margin: float
    normalized_entropy: float
    explanation: str
    ranked: list[tuple[str, float, float]]


def compute_normalized_entropy(distribution: dict[str, float]) -> float:
    
    participant_count = len(distribution)
    if participant_count <= 1:
        return 0.0

    raw_entropy = 0.0

    for probability in distribution.values():
        if probability > 0:
            raw_entropy -= probability * math.log2(probability)

    max_entropy = math.log2(participant_count)
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def evaluate_confidence(
    entries: list[LedgerEntry],
    tracker: ConfidenceTracker,
    current_time: float,
    has_active_conversation_occurred: bool = True,
    meeting_ended: bool = False,
) -> ConfidenceResult:
    
    if not entries:
        tracker.current_status = ConfidenceStatus.UNCERTAIN
        tracker.confirmed_participant_id = None
        return ConfidenceResult(
            status=ConfidenceStatus.UNCERTAIN,
            leading_participant_id=None,
            leading_probability=0.0,
            leading_log_odds=0.0,
            margin=0.0,
            normalized_entropy=0.0,
            explanation="No participants currently in the meeting.",
            ranked=[],
        )

    ranked = get_ranked_participants(entries)
    top_id, top_probability, top_log_odds = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_probability - second_probability

    distribution = {participant_id: probability for participant_id, probability, _ in ranked}
    normalized_entropy = compute_normalized_entropy(distribution)

    entries_by_id = {e.participant_id: e for e in entries}

    max_abs_score = max(abs(e.log_odds_score) for e in entries)
    if (
        current_time >= METADATA_MISMATCH_TIME_THRESHOLD_SECONDS
        and has_active_conversation_occurred
        and max_abs_score < METADATA_MISMATCH_MAX_SCORE_FLOOR
    ):
        tracker.current_status = ConfidenceStatus.METADATA_MISMATCH
        return ConfidenceResult(
            status=ConfidenceStatus.METADATA_MISMATCH,
            leading_participant_id=top_id,
            leading_probability=top_probability,
            leading_log_odds=top_log_odds,
            margin=margin,
            normalized_entropy=normalized_entropy,
            explanation=(
                f"{current_time:.0f}s of active conversation have occurred, but no "
                f"participant has accumulated meaningful evidence in either direction "
                f"(max |log-odds|={max_abs_score:.2f}, below floor "
                f"{METADATA_MISMATCH_MAX_SCORE_FLOOR}). This suggests the external "
                "metadata (calendar invite, candidate name) may not match this meeting, "
                "rather than a failure of the identification logic itself."
            ),
            ranked=ranked,
        )

    if (
        tracker.current_status == ConfidenceStatus.CONFIRMED
        and tracker.confirmed_participant_id is not None
    ):
        confirmed_entry = entries_by_id.get(tracker.confirmed_participant_id)
        if confirmed_entry is not None and not confirmed_entry.is_present:
            return ConfidenceResult(
                status=ConfidenceStatus.TEMPORARILY_ABSENT,
                leading_participant_id=tracker.confirmed_participant_id,
                leading_probability=distribution.get(tracker.confirmed_participant_id, 0.0),
                leading_log_odds=confirmed_entry.log_odds_score,
                margin=margin,
                normalized_entropy=normalized_entropy,
                explanation=(
                    f"Previously confirmed candidate '{tracker.confirmed_participant_id}' "
                    "has left the meeting but remains within the grace window. Status held "
                    "as temporarily absent rather than re-normalizing over the remaining "
                    "participants, which would risk a false promotion of someone else based "
                    "purely on their departure, not new evidence."
                ),
                ranked=ranked,
            )
        if confirmed_entry is None:
            
            tracker.current_status = ConfidenceStatus.UNCERTAIN
            tracker.confirmed_participant_id = None

    if (
        tracker.current_status == ConfidenceStatus.CONFIRMED
        and tracker.confirmed_participant_id == top_id
    ):
       
        still_confirmed = (
            top_log_odds >= CONFIRMED_MAGNITUDE_EXIT_THRESHOLD
            and normalized_entropy <= CONFIRMED_ENTROPY_EXIT_THRESHOLD
        )
        if still_confirmed:
            status = ConfidenceStatus.CONFIRMED
        else:
            status = _classify_quadrant(
                top_log_odds,
                normalized_entropy,
                top_probability=top_probability,
                ranked_size=len(ranked),
            )
    else:
        status = _classify_quadrant(
            top_log_odds,
            normalized_entropy,
            top_probability=top_probability,
            ranked_size=len(ranked),
        )
        if status == ConfidenceStatus.CONFIRMED:
            tracker.confirmed_participant_id = top_id

    tracker.current_status = status

    explanation = _build_explanation(
        status, top_id, top_probability, top_log_odds, margin, normalized_entropy, ranked
    )

    if meeting_ended and status not in (
        ConfidenceStatus.CONFIRMED,
        ConfidenceStatus.TEMPORARILY_ABSENT,
        ConfidenceStatus.METADATA_MISMATCH,
    ):
        status = ConfidenceStatus.UNRESOLVED
        explanation = (
            f"Meeting ended without reaching a confirmed identification. Best guess: "
            f"'{top_id}' (probability={top_probability:.2f}, log-odds={top_log_odds:.2f}), "
            "but this should be treated as a low-confidence guess requiring human review, "
            "not a confirmed result."
        )
        tracker.current_status = status

    return ConfidenceResult(
        status=status,
        leading_participant_id=top_id,
        leading_probability=top_probability,
        leading_log_odds=top_log_odds,
        margin=margin,
        normalized_entropy=normalized_entropy,
        explanation=explanation,
        ranked=ranked,
    )


def _classify_quadrant(
    leading_log_odds: float,
    normalized_entropy: float,
    *,
    top_probability: float = 0.0,
    ranked_size: int = 0,
) -> ConfidenceStatus:
    
    if leading_log_odds <= 0:
        return ConfidenceStatus.UNCERTAIN

    high_magnitude = leading_log_odds >= CONFIDENCE_MAGNITUDE_ENTER_THRESHOLD
    low_entropy = normalized_entropy <= CONFIDENCE_ENTROPY_ENTER_THRESHOLD

    dominant_probability = top_probability >= 0.9
    clear_lead = dominant_probability or ranked_size <= 1
    if clear_lead and (high_magnitude or low_entropy or leading_log_odds >= 1.0):
        return ConfidenceStatus.CONFIRMED

    if high_magnitude and low_entropy:
        return ConfidenceStatus.CONFIRMED
    if high_magnitude and not low_entropy:
        return ConfidenceStatus.PROVISIONAL
    if not high_magnitude and low_entropy:
        return ConfidenceStatus.EARLY_LEAN

    return ConfidenceStatus.UNCERTAIN


def _build_explanation(
    status: ConfidenceStatus,
    top_id: str,
    top_probability: float,
    top_log_odds: float,
    margin: float,
    normalized_entropy: float,
    ranked: list[tuple[str, float, float]],
) -> str:
    if status == ConfidenceStatus.CONFIRMED:
        return (
            f"'{top_id}' confirmed as the candidate (probability={top_probability:.2f}, "
            f"log-odds={top_log_odds:.2f}). Strong evidence has accumulated and the "
            f"ranking is settled (normalized entropy={normalized_entropy:.2f})."
        )
    if status == ConfidenceStatus.PROVISIONAL:
        second_id = ranked[1][0] if len(ranked) > 1 else None
        return (
            f"Strong overall evidence exists (log-odds={top_log_odds:.2f}), but it has not "
            f"broken a tie between the top candidates - '{top_id}' leads "
            f"'{second_id}' by only {margin:.2f} probability. Elimination has narrowed the "
            "pool; confirmation has not yet resolved it. Reporting both rather than "
            "silently picking the higher one."
        )
    if status == ConfidenceStatus.EARLY_LEAN:
        return (
            f"'{top_id}' is ahead of other participants (margin={margin:.2f}), but overall "
            f"evidence is still thin (log-odds={top_log_odds:.2f}). Too early to treat this "
            "as a settled identification."
        )
    if status == ConfidenceStatus.UNCERTAIN:
        return (
            f"Insufficient evidence to identify the candidate with any confidence "
            f"(leading log-odds={top_log_odds:.2f}, normalized entropy={normalized_entropy:.2f} "
            "- belief is spread broadly across participants)."
        )
    return f"Status: {status.value}."