from typing import Optional

from rapidfuzz import fuzz

from app.core.config import (
    AUTOMATED_PARTICIPANT_NAME_KEYWORDS,
    CANDIDATE_COLLISION_GUARD_THRESHOLD,
    CANDIDATE_NAME_MATCH_CATEGORY_CAP,
    CANDIDATE_NAME_MATCH_DELTA,
    CANDIDATE_NAME_MATCH_THRESHOLD,
    INTERVIEWER_NAME_MATCH_THRESHOLD,
    KNOWN_INTERVIEWER_CATEGORY_CAP,
    KNOWN_INTERVIEWER_ELIMINATION_DELTA,
    SILENCE_AUTOMATED_NAME_AMPLIFIER_DELTA,
    SILENCE_CATEGORY_CAP,
    SILENCE_ELIMINATION_INCREMENT_DELTA,
    SILENCE_INITIAL_THRESHOLD_SECONDS,
    SILENCE_POSITIVE_EVIDENCE_DAMPENING_FACTOR,
)
from app.core.ledger import apply_scored_delta
from app.core.registry import LedgerEntry
from app.models.schemas import ExternalMetadata


def _name_similarity(a: str, b: str) -> float:
    
    if not a or not b:
        return 0.0
    return max(
        fuzz.token_sort_ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    )


def _collect_known_interviewer_emails(external_metadata: ExternalMetadata) -> set[str]:
    emails: set[str] = set()
    if external_metadata.calendar_invite is not None:
        for attendee in external_metadata.calendar_invite.attendees:
            if attendee.email:
                emails.add(attendee.email.strip().lower())

    candidate_email = (external_metadata.candidate_email or "").strip().lower()
    if candidate_email:
        emails.discard(candidate_email)

    return emails


def eliminate_by_known_interviewer(
    entry: LedgerEntry,
    display_name: str,
    participant_email: Optional[str],
    external_metadata: ExternalMetadata,
    event_sequence_id: Optional[int] = None,
) -> None:
    
    interviewer_emails = _collect_known_interviewer_emails(external_metadata)
    candidate_email = (external_metadata.candidate_email or "").strip().lower()
    normalized_participant_email = (participant_email or "").strip().lower()

    if normalized_participant_email:
        if normalized_participant_email in interviewer_emails and normalized_participant_email != candidate_email:
            apply_scored_delta(
                entry,
                category="known_interviewer_email",
                delta=KNOWN_INTERVIEWER_ELIMINATION_DELTA,
                reason=(
                    f"Participant email '{normalized_participant_email}' matches a known "
                    "interviewer email sourced from calendar invite attendees."
                ),
                event_sequence_id=event_sequence_id,
                category_cap=KNOWN_INTERVIEWER_CATEGORY_CAP,
            )
            return

    best_interviewer_name: Optional[str] = None
    best_interviewer_similarity = 0.0
    for interviewer_name in external_metadata.interviewer_names:
        similarity = _name_similarity(display_name, interviewer_name)
        if similarity > best_interviewer_similarity:
            best_interviewer_similarity = similarity
            best_interviewer_name = interviewer_name

    if best_interviewer_similarity >= INTERVIEWER_NAME_MATCH_THRESHOLD:
        
        candidate_similarity = _name_similarity(display_name, external_metadata.candidate_name)

        if candidate_similarity >= CANDIDATE_COLLISION_GUARD_THRESHOLD:
            apply_scored_delta(
                entry,
                category="known_interviewer_suppressed",
                delta=0.0,
                reason=(
                    f"Display name '{display_name}' matches known interviewer "
                    f"'{best_interviewer_name}' (similarity={best_interviewer_similarity:.0f}), "
                    f"but is ALSO similar to the candidate name (similarity={candidate_similarity:.0f}). "
                    "Elimination suppressed to avoid false-eliminating the real candidate; "
                    "deferring to behavioral evidence from later steps."
                ),
                event_sequence_id=event_sequence_id,
            )
            return

        apply_scored_delta(
            entry,
            category="known_interviewer_name",
            delta=KNOWN_INTERVIEWER_ELIMINATION_DELTA,
            reason=(
                f"Display name '{display_name}' matches known interviewer "
                f"'{best_interviewer_name}' (similarity={best_interviewer_similarity:.0f})."
            ),
            event_sequence_id=event_sequence_id,
            category_cap=KNOWN_INTERVIEWER_CATEGORY_CAP,
        )
        return

    apply_scored_delta(
        entry,
        category="calendar_lookup",
        delta=0.0,
        reason=(
            "No matching entry found in interviewer_names or calendar_invite.attendees "
            "for this participant. Identification will rely entirely on live "
            "behavioral evidence from later steps."
        ),
        event_sequence_id=event_sequence_id,
    )


def confirm_by_candidate_name_match(
    entry: LedgerEntry,
    display_name: str,
    external_metadata: ExternalMetadata,
    event_sequence_id: Optional[int] = None,
) -> None:
    
    similarity = _name_similarity(display_name, external_metadata.candidate_name)

    if similarity >= CANDIDATE_NAME_MATCH_THRESHOLD:
        apply_scored_delta(
            entry,
            category="candidate_name_match",
            delta=CANDIDATE_NAME_MATCH_DELTA,
            reason=(
                f"Display name '{display_name}' closely matches candidate name "
                f"'{external_metadata.candidate_name}' (similarity={similarity:.0f})."
            ),
            event_sequence_id=event_sequence_id,
            category_cap=CANDIDATE_NAME_MATCH_CATEGORY_CAP,
        )
        return

    apply_scored_delta(
        entry,
        category="candidate_name_match",
        delta=0.0,
        reason=(
            f"Display name '{display_name}' does not meaningfully resemble candidate "
            f"name '{external_metadata.candidate_name}' (similarity={similarity:.0f}). "
            "Absence of a match is neutral, not disqualifying."
        ),
        event_sequence_id=event_sequence_id,
    )


def looks_like_automated_participant_name(display_name: str) -> Optional[str]:
    
    normalized_name = display_name.lower()
    return next(
        (keyword for keyword in AUTOMATED_PARTICIPANT_NAME_KEYWORDS if keyword in normalized_name),
        None,
    )


def eliminate_by_silence(
    entry: LedgerEntry,
    current_time: float,
    participant_total_speaking_seconds: float,
    other_participants_speaking_seconds: float,
    has_conversational_exchange_occurred: bool,
    is_screen_sharing_active: bool,
    automated_name_keyword: Optional[str] = None,
    event_sequence_id: Optional[int] = None,
) -> None:
    

    if participant_total_speaking_seconds > 0:
        apply_scored_delta(
            entry,
            category="silence",
            delta=0.0,
            reason="Participant has recorded speaking activity; silence-elimination does not apply.",
            event_sequence_id=event_sequence_id,
        )
        return

    if is_screen_sharing_active:
        apply_scored_delta(
            entry,
            category="silence",
            delta=0.0,
            reason=(
                "Participant is silent but is actively screen-sharing -- suppressed per "
                "Step 4: silence during active sharing (e.g. live-coding) is not suspicious."
            ),
            event_sequence_id=event_sequence_id,
        )
        return

    if not has_conversational_exchange_occurred:
        apply_scored_delta(
            entry,
            category="silence",
            delta=0.0,
            reason=(
                "No conversational exchange has occurred yet in the meeting -- likely still "
                "in opening remarks. Silence is not evaluated until a genuine exchange begins."
            ),
            event_sequence_id=event_sequence_id,
        )
        return

    if other_participants_speaking_seconds <= 0:
        apply_scored_delta(
            entry,
            category="silence",
            delta=0.0,
            reason=(
                "All other participants are also currently silent (shared pause or "
                "connectivity stall) -- no differentiating signal, elimination suppressed."
            ),
            event_sequence_id=event_sequence_id,
        )
        return

    elapsed_since_join = current_time - entry.joined_at
    if elapsed_since_join < SILENCE_INITIAL_THRESHOLD_SECONDS:
        apply_scored_delta(
            entry,
            category="silence",
            delta=0.0,
            reason=(
                f"Silent for {elapsed_since_join:.0f}s since joining, still under the "
                f"{SILENCE_INITIAL_THRESHOLD_SECONDS:.0f}s initial threshold."
            ),
            event_sequence_id=event_sequence_id,
        )
        return

    dampening = (
        SILENCE_POSITIVE_EVIDENCE_DAMPENING_FACTOR if entry.log_odds_score > 0 else 1.0
    )
    effective_delta = SILENCE_ELIMINATION_INCREMENT_DELTA * dampening

    dampening_note = (
        f" Dampened by {dampening:.1f}x since participant already holds net-positive "
        f"evidence (score={entry.log_odds_score:.2f})."
        if dampening != 1.0
        else ""
    )

    apply_scored_delta(
        entry,
        category="silence",
        delta=effective_delta,
        reason=(
            f"Zero speaking activity for {elapsed_since_join:.0f}s while other participants "
            f"actively spoke ({other_participants_speaking_seconds:.0f}s combined)."
            + dampening_note
        ),
        event_sequence_id=event_sequence_id,
        category_cap=SILENCE_CATEGORY_CAP,
    )

    if automated_name_keyword is not None:
        apply_scored_delta(
            entry,
            category="silence",
            delta=SILENCE_AUTOMATED_NAME_AMPLIFIER_DELTA,
            reason=(
                f"Additionally, display name matches automated-participant pattern "
                f"'{automated_name_keyword}', corroborating the silence-based signal."
            ),
            event_sequence_id=event_sequence_id,
            category_cap=SILENCE_CATEGORY_CAP,
        )