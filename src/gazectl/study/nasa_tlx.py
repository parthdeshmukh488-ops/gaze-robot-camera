"""NASA-TLX workload scoring (Hart and Staveland, 1988).

Six subscales, each rated 0-100. Two scoring modes:

- **Raw TLX** — the unweighted mean of the six. Common, defensible, and what
  most HCI papers now report.
- **Weighted TLX** — the participant also does 15 pairwise comparisons saying
  which of each pair contributed more to their workload; the number of times a
  subscale wins becomes its weight (0-5, summing to 15), and the score is the
  weighted mean.

The weighting procedure roughly doubles how long the questionnaire takes, and
the evidence that it improves sensitivity is weak, so Raw TLX is the default
here. The weighted version is implemented because the comparison is part of the
study design, not because it is recommended.

**Performance is reverse-coded.** Its anchors run from "perfect" to "failure",
so a participant who did well marks a *low* number. Every other subscale runs
low-to-high on demand. Mixing them up inverts a sixth of the score and it is
the single most common NASA-TLX error, so :func:`raw_tlx` takes performance in
its native "perfect=0" form and handles it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

#: Subscale order used throughout this module.
SUBSCALES = (
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
)

N_SUBSCALES = len(SUBSCALES)
N_PAIRS = len(list(combinations(range(N_SUBSCALES), 2)))  # 15

#: The 15 pairwise comparisons, in the canonical order.
PAIRS = tuple(combinations(SUBSCALES, 2))


@dataclass(frozen=True)
class TLXResult:
    """A scored NASA-TLX response."""

    score: float  # 0-100, higher means more workload
    subscale_scores: dict
    weights: dict | None = None

    @property
    def weighted(self) -> bool:
        return self.weights is not None

    def __str__(self) -> str:
        mode = "weighted" if self.weighted else "raw"
        return f"NASA-TLX {self.score:.1f} ({mode})"


def _validate_ratings(ratings: dict) -> np.ndarray:
    missing = set(SUBSCALES) - set(ratings)
    if missing:
        raise ValueError(f"missing subscale ratings: {sorted(missing)}")

    unexpected = set(ratings) - set(SUBSCALES)
    if unexpected:
        raise ValueError(f"unexpected keys: {sorted(unexpected)}")

    values = np.array([float(ratings[name]) for name in SUBSCALES])
    if np.any(values < 0) or np.any(values > 100):
        raise ValueError("every subscale rating must lie in 0..100")
    return values


def raw_tlx(ratings: dict) -> TLXResult:
    """Unweighted NASA-TLX.

    Args:
        ratings: all six subscales, 0-100. ``performance`` is given in its
            native anchoring, where 0 means perfect and 100 means failure.

    Returns:
        The score, where higher always means more workload.
    """
    values = _validate_ratings(ratings)
    return TLXResult(
        score=float(values.mean()),
        subscale_scores={name: float(v) for name, v in zip(SUBSCALES, values, strict=True)},
    )


def weights_from_comparisons(winners) -> dict:
    """Turn the 15 pairwise comparisons into subscale weights.

    Args:
        winners: 15 subscale names, one per pair in :data:`PAIRS` order, each
            naming the member of that pair the participant judged the larger
            contributor.

    Returns:
        A weight per subscale, summing to 15.
    """
    winners = list(winners)
    if len(winners) != N_PAIRS:
        raise ValueError(f"need exactly {N_PAIRS} comparisons, got {len(winners)}")

    counts = {name: 0 for name in SUBSCALES}
    for chosen, pair in zip(winners, PAIRS, strict=True):
        if chosen not in pair:
            raise ValueError(f"{chosen!r} is not a member of the pair {pair}")
        counts[chosen] += 1

    return counts


def weighted_tlx(ratings: dict, winners) -> TLXResult:
    """Weighted NASA-TLX, using the participant's own pairwise comparisons."""
    values = _validate_ratings(ratings)
    weights = weights_from_comparisons(winners)
    weight_vector = np.array([weights[name] for name in SUBSCALES], dtype=float)

    # The weights always sum to 15 by construction, so this cannot divide by
    # zero; the assertion documents the invariant rather than guarding it.
    assert weight_vector.sum() == N_PAIRS

    return TLXResult(
        score=float(np.dot(values, weight_vector) / N_PAIRS),
        subscale_scores={name: float(v) for name, v in zip(SUBSCALES, values, strict=True)},
        weights=weights,
    )
