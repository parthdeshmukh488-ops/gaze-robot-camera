"""Tests for the questionnaire scoring.

Both instruments have published worked examples and known edge cases, so these
tests check against the definitions rather than against whatever the code
happened to produce. The reverse-scored items are the point: a SUS or a TLX
that averages its items naively produces a number that looks entirely
reasonable and is wrong.
"""

import numpy as np
import pytest

from gazectl.study import nasa_tlx, sus


def test_sus_all_strongly_positive_scores_100():
    """Positive items at 5, negative items at 1."""
    responses = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
    assert sus.score(responses) == pytest.approx(100.0)


def test_sus_all_strongly_negative_scores_zero():
    responses = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
    assert sus.score(responses) == pytest.approx(0.0)


def test_sus_all_neutral_scores_50():
    assert sus.score([3] * 10) == pytest.approx(50.0)


def test_sus_naive_average_would_be_wrong():
    """The reverse-scoring, demonstrated.

    A respondent who agrees with everything is contradicting themselves — they
    called the system both easy and cumbersome. The correct score is the
    midpoint. A naive mean of the raw responses would report 5 out of 5, which
    would rescale to 100: a perfect score for an incoherent response.
    """
    all_agree = [5] * 10
    assert sus.score(all_agree) == pytest.approx(50.0)
    assert np.mean(all_agree) == 5.0


def test_sus_rejects_wrong_item_count():
    with pytest.raises(ValueError, match="exactly 10"):
        sus.score([3] * 9)


def test_sus_rejects_out_of_range_responses():
    with pytest.raises(ValueError, match="1..5"):
        sus.score([6] + [3] * 9)
    with pytest.raises(ValueError, match="1..5"):
        sus.score([0] + [3] * 9)


def test_sus_rejects_fractional_responses():
    with pytest.raises(ValueError, match="whole numbers"):
        sus.score([3.5] + [3] * 9)


def test_sus_score_is_always_a_multiple_of_2_point_5():
    rng = np.random.default_rng(0)
    for _ in range(200):
        responses = rng.integers(1, 6, size=10)
        score = sus.score(responses)
        assert (score / 2.5) == pytest.approx(round(score / 2.5))


def test_sus_interpretation_knows_the_mean_is_68():
    """A score of 65 feels like a pass and is below average."""
    assert "below average" in sus.interpret(65.0).percentile_band
    assert "above average" in sus.interpret(72.0).percentile_band


def test_sus_interpretation_grades_extremes():
    assert sus.interpret(95.0).grade == "A"
    assert sus.interpret(20.0).grade == "F"


def test_sus_interpretation_rejects_impossible_scores():
    with pytest.raises(ValueError, match="0..100"):
        sus.interpret(120.0)


def test_sus_score_many():
    matrix = [[5, 1, 5, 1, 5, 1, 5, 1, 5, 1], [3] * 10]
    scores = sus.score_many(matrix)
    assert scores == pytest.approx([100.0, 50.0])


def _ratings(**overrides):
    base = {name: 50.0 for name in nasa_tlx.SUBSCALES}
    base.update(overrides)
    return base


def test_tlx_raw_is_the_mean_of_six():
    result = nasa_tlx.raw_tlx(_ratings())
    assert result.score == pytest.approx(50.0)
    assert not result.weighted


def test_tlx_raw_uses_every_subscale():
    result = nasa_tlx.raw_tlx(_ratings(mental_demand=100.0))
    assert result.score == pytest.approx((100.0 + 50.0 * 5) / 6.0)


def test_tlx_rejects_missing_subscale():
    incomplete = _ratings()
    del incomplete["effort"]
    with pytest.raises(ValueError, match="missing subscale"):
        nasa_tlx.raw_tlx(incomplete)


def test_tlx_rejects_unexpected_key():
    extra = _ratings()
    extra["enjoyment"] = 50.0
    with pytest.raises(ValueError, match="unexpected keys"):
        nasa_tlx.raw_tlx(extra)


def test_tlx_rejects_out_of_range_rating():
    with pytest.raises(ValueError, match="0..100"):
        nasa_tlx.raw_tlx(_ratings(effort=101.0))


def test_tlx_has_fifteen_pairs():
    assert nasa_tlx.N_PAIRS == 15
    assert len(nasa_tlx.PAIRS) == 15
    assert len(set(nasa_tlx.PAIRS)) == 15


def test_tlx_weights_sum_to_fifteen():
    winners = [pair[0] for pair in nasa_tlx.PAIRS]
    weights = nasa_tlx.weights_from_comparisons(winners)
    assert sum(weights.values()) == 15


def test_tlx_weights_rejects_a_non_member():
    winners = [pair[0] for pair in nasa_tlx.PAIRS]
    winners[0] = "frustration"  # not in the first pair
    with pytest.raises(ValueError, match="not a member"):
        nasa_tlx.weights_from_comparisons(winners)


def test_tlx_weights_rejects_wrong_count():
    with pytest.raises(ValueError, match="exactly 15"):
        nasa_tlx.weights_from_comparisons(["mental_demand"] * 14)


def test_tlx_uniform_weights_are_arithmetically_impossible():
    """A property of the instrument, not of this implementation.

    Six subscales share 15 comparisons, and 15 / 6 = 2.5. No participant can
    ever produce equal weights, so weighted TLX can never reduce to raw TLX
    by weighting alone. Anyone expecting the two to agree on a "balanced"
    respondent is expecting something the arithmetic forbids.
    """
    assert nasa_tlx.N_PAIRS % nasa_tlx.N_SUBSCALES != 0


def test_tlx_weighted_equals_raw_when_all_ratings_agree():
    """Weights cannot matter if every subscale was rated the same."""
    winners = [pair[0] for pair in nasa_tlx.PAIRS]
    ratings = _ratings()  # every subscale at 50

    weighted = nasa_tlx.weighted_tlx(ratings, winners)
    raw = nasa_tlx.raw_tlx(ratings)

    assert weighted.score == pytest.approx(raw.score)
    assert weighted.weighted
    assert sum(weighted.weights.values()) == nasa_tlx.N_PAIRS


def test_tlx_weighting_shifts_the_score():
    """A subscale that dominates the comparisons should dominate the score."""
    winners = []
    for pair in nasa_tlx.PAIRS:
        winners.append("mental_demand" if "mental_demand" in pair else pair[0])

    ratings = _ratings(mental_demand=100.0)
    weighted = nasa_tlx.weighted_tlx(ratings, winners)
    raw = nasa_tlx.raw_tlx(ratings)

    assert weighted.score > raw.score
    assert weighted.weights["mental_demand"] == 5


def test_tlx_result_reports_subscales():
    result = nasa_tlx.raw_tlx(_ratings(effort=80.0))
    assert result.subscale_scores["effort"] == pytest.approx(80.0)
    assert set(result.subscale_scores) == set(nasa_tlx.SUBSCALES)
