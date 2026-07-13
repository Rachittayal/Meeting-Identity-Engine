from typing import Optional

from app.core.registry import LedgerEntry


def apply_scored_delta(
    entry: LedgerEntry,
    category: str,
    delta: float,
    reason: str,
    event_sequence_id: Optional[int] = None,
    category_cap: Optional[float] = None,
) -> float:
    current_category_total = entry.category_contributions.get(category, 0.0)

    if category_cap is not None:
        
        proposed_total = current_category_total + delta
        clamped_total = max(-category_cap, min(category_cap, proposed_total))
        applied_delta = clamped_total - current_category_total
    else:
        applied_delta = delta

    entry.log_odds_score += applied_delta
    entry.category_contributions[category] = current_category_total + applied_delta

    entry.evidence_log.append(
        {
            "event_sequence_id": event_sequence_id,
            "category": category,
            "requested_delta": delta,
            "applied_delta": applied_delta,
            "reason": reason,
            "running_score": entry.log_odds_score,
        }
    )
    return applied_delta