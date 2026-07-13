import os
from typing import Tuple


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"SHERLOCK_{name}", default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"SHERLOCK_{name}", default))

MAX_GAP_FOR_TURN_MERGE_SECONDS: float = env_float("MAX_GAP_TURN_MERGE", 1.5)

LATE_JOIN_PENALTY_LOG_ODDS: float = env_float("LATE_JOIN_PENALTY", -0.3)

LATE_JOIN_THRESHOLD_SECONDS: float = env_float("LATE_JOIN_THRESHOLD", 180.0)

LEAVE_GRACE_WINDOW_SECONDS: float = env_float("LEAVE_GRACE_WINDOW", 600.0)

INTERVIEWER_NAME_MATCH_THRESHOLD: float = env_float("INTERVIEWER_NAME_THRESHOLD", 90.0)

CANDIDATE_COLLISION_GUARD_THRESHOLD: float = env_float("CANDIDATE_COLLISION_GUARD", 78.0)

KNOWN_INTERVIEWER_ELIMINATION_DELTA: float = env_float("INTERVIEWER_DELTA", -6.0)

KNOWN_INTERVIEWER_CATEGORY_CAP: float = env_float("INTERVIEWER_CAP", 8.0)

CANDIDATE_NAME_MATCH_THRESHOLD: float = env_float("CANDIDATE_NAME_THRESHOLD", 82.0)

CANDIDATE_NAME_MATCH_DELTA: float = env_float("CANDIDATE_NAME_DELTA", 1.2)

CANDIDATE_NAME_MATCH_CATEGORY_CAP: float = env_float("CANDIDATE_NAME_CAP", 3.0)

_AUTOMATED_KEYWORDS_DEFAULT = (
    "recorder,notetaker,note taker,note-taker,transcription bot,meeting bot,"
    "otter.ai,otter pilot,fireflies,gong.io,chorus.ai,fathom"
)
AUTOMATED_PARTICIPANT_NAME_KEYWORDS: Tuple[str, ...] = tuple(
    kw.strip()
    for kw in os.environ.get("SHERLOCK_AUTOMATED_KEYWORDS", _AUTOMATED_KEYWORDS_DEFAULT).split(",")
    if kw.strip()
)

SILENCE_INITIAL_THRESHOLD_SECONDS: float = env_float("SILENCE_INITIAL_THRESHOLD", 300.0)  # 5 min

SILENCE_CHECK_INTERVAL_SECONDS: float = env_float("SILENCE_CHECK_INTERVAL", 300.0)  # 5 min

SILENCE_ELIMINATION_INCREMENT_DELTA: float = env_float("SILENCE_INCREMENT", -1.0)

SILENCE_CATEGORY_CAP: float = env_float("SILENCE_CAP", 3.0)

SILENCE_POSITIVE_EVIDENCE_DAMPENING_FACTOR: float = env_float("SILENCE_DAMPENING", 0.5)

SILENCE_AUTOMATED_NAME_AMPLIFIER_DELTA: float = env_float("SILENCE_AUTOMATED_AMPLIFIER", -1.0)