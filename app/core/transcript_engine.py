import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from app.core.config import (
    CANDIDATE_NARRATIVE_BASE_DELTA,
    CANDIDATE_NARRATIVE_ESTABLISHED_DELTA,
    CANDIDATE_ROLE_CATEGORY_CAP,
    EVALUATIVE_QUESTION_BASE_DELTA,
    EVALUATIVE_QUESTION_ESTABLISHED_DELTA,
    INTERVIEWER_ROLE_CATEGORY_CAP,
    MIN_ASR_CONFIDENCE_FOR_CLASSIFICATION,
    MIN_CLASSIFIER_CONFIDENCE,
    MIN_INFORMATIVE_WORD_COUNT,
    PATTERN_ESTABLISHMENT_TURN_COUNT,
    SPOKEN_INTRODUCTION_CATEGORY_CAP,
    SPOKEN_INTRODUCTION_DELTA,
    TURN_TAKING_AFTER_INTERVIEWER_DELTA,
    TURN_TAKING_CATEGORY_CAP,
)
from app.core.ledger import apply_scored_delta
from app.core.registry import ConversationalTurn, LedgerEntry
from app.models.schemas import ExternalMetadata


class TurnRoleLabel(str, Enum):
    EVALUATIVE_QUESTION = "evaluative_question"
    INFORMATIONAL_QUESTION = "informational_question"
    CANDIDATE_NARRATIVE = "candidate_narrative"
    SELF_INTRODUCTION = "self_introduction"
    NONE = "none"


@dataclass
class TurnClassification:
    
    label: TurnRoleLabel
    confidence: float
    applied_delta: float = 0.0


@dataclass
class ParticipantTurnHistory:

    evaluative_question_count: int = 0
    candidate_narrative_count: int = 0
    total_classified_count: int = 0


def is_segment_informative(turn: ConversationalTurn) -> bool:
    return len(turn.text.split()) >= MIN_INFORMATIVE_WORD_COUNT


def _default_heuristic_classifier(
    turn: ConversationalTurn,
    preceding_context: list[ConversationalTurn],
    external_metadata: ExternalMetadata,
) -> TurnClassification:
    
    text_lower = turn.text.lower()
    has_question_mark = "?" in turn.text

    self_intro_pattern = re.search(
        r"\b(this is|speaking|that'?s me|yes,? this is)\b", text_lower
    )
    preceding_mentions_candidate = any(
        external_metadata.candidate_name.split()[0].lower() in t.text.lower()
        for t in preceding_context
    ) if external_metadata.candidate_name else False

    if self_intro_pattern and preceding_mentions_candidate:
        return TurnClassification(TurnRoleLabel.SELF_INTRODUCTION, confidence=0.85)

    evaluative_pattern = re.search(
        r"\b(tell me about|walk (me|us|everyone) through|how would you|why did you|"
        r"what was your approach|describe a time|give an example|"
        r"can you explain|could you (walk|tell))\b",
        text_lower,
    )
    if evaluative_pattern and has_question_mark:
        return TurnClassification(TurnRoleLabel.EVALUATIVE_QUESTION, confidence=0.80)

    informational_pattern = re.search(
        r"\b(what'?s the|what is the|do you have (any )?questions|"
        r"how big is|tech stack|tell me more about (the|your) (role|team|company))\b",
        text_lower,
    )
    if informational_pattern and has_question_mark:
        return TurnClassification(TurnRoleLabel.INFORMATIONAL_QUESTION, confidence=0.75)

    narrative_pattern = re.search(
        r"\b(i|we) (led|built|worked|used|implemented|managed|designed|created|"
        r"developed|migrated|owned|shipped)\b",
        text_lower,
    )
    if narrative_pattern:
        return TurnClassification(TurnRoleLabel.CANDIDATE_NARRATIVE, confidence=0.80)

    return TurnClassification(TurnRoleLabel.NONE, confidence=0.50)


TurnClassifier = Callable[[ConversationalTurn, list[ConversationalTurn], ExternalMetadata], TurnClassification]


def _scaled_delta(base: float, established: float, occurrence_count: int) -> float:
    
    progress = min(1.0, occurrence_count / PATTERN_ESTABLISHMENT_TURN_COUNT)
    return base + (established - base) * progress


def process_turn(
    turn: ConversationalTurn,
    preceding_context: list[ConversationalTurn],
    entry: LedgerEntry,
    external_metadata: ExternalMetadata,
    history: ParticipantTurnHistory,
    previous_turn_speaker_is_known_interviewer: bool = False,
    classifier: TurnClassifier = _default_heuristic_classifier,
    event_sequence_id: Optional[int] = None,
) -> Optional[TurnClassification]:
    if not is_segment_informative(turn):
        apply_scored_delta(
            entry, category="transcript_role", delta=0.0,
            reason=f"Turn too short/uninformative ({len(turn.text.split())} words) to classify.",
            event_sequence_id=event_sequence_id,
        )
        return None

    if turn.min_confidence is not None and turn.min_confidence < MIN_ASR_CONFIDENCE_FOR_CLASSIFICATION:
        apply_scored_delta(
            entry, category="transcript_role", delta=0.0,
            reason=(
                f"ASR confidence {turn.min_confidence:.2f} below floor "
                f"{MIN_ASR_CONFIDENCE_FOR_CLASSIFICATION} - transcription may be unreliable, "
                "discarded before semantic classification."
            ),
            event_sequence_id=event_sequence_id,
        )
        return None

    classification = classifier(turn, preceding_context, external_metadata)

    if classification.confidence < MIN_CLASSIFIER_CONFIDENCE:
        apply_scored_delta(
            entry, category="transcript_role", delta=0.0,
            reason=(
                f"Classifier confidence {classification.confidence:.2f} below floor "
                f"{MIN_CLASSIFIER_CONFIDENCE} for label '{classification.label.value}' - "
                "ambiguous turn discarded, not netted to zero."
            ),
            event_sequence_id=event_sequence_id,
        )
        return None

    history.total_classified_count += 1

    if classification.label == TurnRoleLabel.EVALUATIVE_QUESTION:
        history.evaluative_question_count += 1
        delta = _scaled_delta(
            EVALUATIVE_QUESTION_BASE_DELTA, EVALUATIVE_QUESTION_ESTABLISHED_DELTA,
            history.evaluative_question_count,
        )
        applied = apply_scored_delta(
            entry, category="transcript_role", delta=delta,
            reason=(
                f"Turn classified as evaluative question (classifier_confidence="
                f"{classification.confidence:.2f}), occurrence #{history.evaluative_question_count} "
                f"for this participant. Text: \"{turn.text[:80]}\""
            ),
            event_sequence_id=event_sequence_id,
            category_cap=INTERVIEWER_ROLE_CATEGORY_CAP,
        )
        classification.applied_delta = applied

    elif classification.label == TurnRoleLabel.CANDIDATE_NARRATIVE:
        history.candidate_narrative_count += 1
        delta = _scaled_delta(
            CANDIDATE_NARRATIVE_BASE_DELTA, CANDIDATE_NARRATIVE_ESTABLISHED_DELTA,
            history.candidate_narrative_count,
        )
        applied = apply_scored_delta(
            entry, category="transcript_role", delta=delta,
            reason=(
                f"Turn classified as candidate narrative (classifier_confidence="
                f"{classification.confidence:.2f}), occurrence #{history.candidate_narrative_count} "
                f"for this participant. Text: \"{turn.text[:80]}\""
            ),
            event_sequence_id=event_sequence_id,
            category_cap=CANDIDATE_ROLE_CATEGORY_CAP,
        )
        classification.applied_delta = applied

    elif classification.label == TurnRoleLabel.SELF_INTRODUCTION:
        apply_scored_delta(
            entry, category="spoken_introduction", delta=SPOKEN_INTRODUCTION_DELTA,
            reason=(
                f"Turn classified as spoken self-introduction responding to candidate name "
                f"being addressed in preceding context (classifier_confidence="
                f"{classification.confidence:.2f})."
            ),
            event_sequence_id=event_sequence_id,
            category_cap=SPOKEN_INTRODUCTION_CATEGORY_CAP,
        )

    elif classification.label == TurnRoleLabel.INFORMATIONAL_QUESTION:
        
        apply_scored_delta(
            entry, category="transcript_role", delta=0.0,
            reason=(
                f"Turn classified as informational question (classifier_confidence="
                f"{classification.confidence:.2f}) - does not feed interviewer-role evidence, "
                "since candidates ask informational questions too."
            ),
            event_sequence_id=event_sequence_id,
        )

    else:
        apply_scored_delta(
            entry, category="transcript_role", delta=0.0,
            reason="Turn classified as ambiguous/small-talk (label=none) - no evidence contributed.",
            event_sequence_id=event_sequence_id,
        )

    if previous_turn_speaker_is_known_interviewer and classification.label in (
        TurnRoleLabel.CANDIDATE_NARRATIVE, TurnRoleLabel.SELF_INTRODUCTION,
    ):
        apply_scored_delta(
            entry, category="turn_taking_pattern", delta=TURN_TAKING_AFTER_INTERVIEWER_DELTA,
            reason=(
                "This turn immediately follows a turn from a participant already identified "
                "as a known interviewer, and reads as an answer rather than an interruption - "
                "weak corroborating signal."
            ),
            event_sequence_id=event_sequence_id,
            category_cap=TURN_TAKING_CATEGORY_CAP,
        )

    return classification


def is_known_interviewer(entry: LedgerEntry) -> bool:
    
    return (
        entry.category_contributions.get("known_interviewer_email", 0.0) < 0
        or entry.category_contributions.get("known_interviewer_name", 0.0) < 0
    )