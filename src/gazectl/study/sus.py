"""System Usability Scale scoring (Brooke, 1996).

Ten items on a 1-5 agreement scale, alternating positive and negative wording.
The scoring is not an average: odd items contribute ``response - 1``, even
items contribute ``5 - response``, and the total is multiplied by 2.5 to land
on 0-100.

Two things about that 0-100 number that are routinely got wrong, and which this
module refuses to let a caller get wrong:

**It is not a percentage.** A SUS of 70 does not mean 70% of anything. It is a
point on an arbitrary scale whose only meaning comes from the distribution of
other systems' scores.

**The mean is 68, not 50.** Scoring 65 feels like a pass and is in fact below
average. :func:`interpret` exists so that a score is never reported without
that context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ITEMS = 10
MIN_RESPONSE = 1
MAX_RESPONSE = 5

#: The SUS items, in order. Odd indices (1-based) are positively worded.
ITEMS = (
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
)


@dataclass(frozen=True)
class SUSResult:
    """A scored SUS response."""

    score: float  # 0-100
    grade: str
    adjective: str
    percentile_band: str

    def __str__(self) -> str:
        return f"SUS {self.score:.1f} ({self.adjective}, grade {self.grade})"


def score(responses) -> float:
    """Score one participant's SUS responses.

    Args:
        responses: 10 integers in 1..5, in item order.

    Returns:
        The SUS score in 0..100.
    """
    responses = np.asarray(responses, dtype=float).ravel()

    if responses.size != N_ITEMS:
        raise ValueError(f"SUS needs exactly {N_ITEMS} responses, got {responses.size}")
    if np.any(responses < MIN_RESPONSE) or np.any(responses > MAX_RESPONSE):
        raise ValueError(f"responses must lie in {MIN_RESPONSE}..{MAX_RESPONSE}")
    if not np.all(np.equal(np.mod(responses, 1), 0)):
        raise ValueError("SUS responses must be whole numbers")

    odd = responses[0::2] - 1.0  # items 1, 3, 5, 7, 9
    even = 5.0 - responses[1::2]  # items 2, 4, 6, 8, 10
    return float((odd.sum() + even.sum()) * 2.5)


def interpret(sus_score: float) -> SUSResult:
    """Attach the standard interpretive labels to a SUS score.

    Grades and adjectives follow Bangor, Kortum and Miller (2009) and Sauro's
    curved grading. The bands are coarse on purpose — SUS does not resolve
    finely enough to justify anything narrower, and a two-point difference
    between conditions is noise.
    """
    if not 0.0 <= sus_score <= 100.0:
        raise ValueError(f"SUS score must lie in 0..100, got {sus_score}")

    if sus_score >= 84.1:
        grade, adjective, band = "A", "best imaginable", "top 5%"
    elif sus_score >= 80.8:
        grade, adjective, band = "A-", "excellent", "top 10%"
    elif sus_score >= 78.9:
        grade, adjective, band = "B+", "good", "top 20%"
    elif sus_score >= 71.1:
        grade, adjective, band = "B", "good", "above average"
    elif sus_score >= 68.0:
        grade, adjective, band = "C+", "okay", "just above average"
    elif sus_score >= 62.7:
        grade, adjective, band = "C", "okay", "just below average"
    elif sus_score >= 51.7:
        grade, adjective, band = "D", "poor", "bottom 35%"
    else:
        grade, adjective, band = "F", "awful", "bottom 15%"

    return SUSResult(
        score=float(sus_score), grade=grade, adjective=adjective, percentile_band=band
    )


def score_many(responses_matrix) -> np.ndarray:
    """Score a (participants, 10) matrix. Returns one score per participant."""
    matrix = np.atleast_2d(np.asarray(responses_matrix, dtype=float))
    return np.array([score(row) for row in matrix])
