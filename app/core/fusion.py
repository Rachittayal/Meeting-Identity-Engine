import math

from app.core.registry import LedgerEntry


def compute_probability_distribution(entries: list[LedgerEntry]) -> dict[str, float]:
    
    if not entries:
        return {}

    if len(entries) == 1:
        return {entries[0].participant_id: 1.0}

    max_score = max(e.log_odds_score for e in entries)
    exp_scores = {e.participant_id: math.exp(e.log_odds_score - max_score) for e in entries}
    total = sum(exp_scores.values())

    return {participant_id: value / total for participant_id, value in exp_scores.items()}


def get_ranked_participants(entries: list[LedgerEntry]) -> list[tuple[str, float, float]]:
    
    distribution = compute_probability_distribution(entries)
    score_by_id = {e.participant_id: e.log_odds_score for e in entries}

    ranked = [
        (participant_id, probability, score_by_id[participant_id])
        for participant_id, probability in distribution.items()
    ]
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked