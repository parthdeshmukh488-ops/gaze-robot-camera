"""Tests for I-VT fixation detection.

Built around synthetic traces where the right answer is known by construction:
a fixation is a cluster of samples at one place, a saccade is a fast run
between two places, and the detector either recovers that structure or it does
not.
"""

import numpy as np
import pytest

from gazectl.filters import angular_velocity, detect_fixations
from gazectl.geometry import ScreenGeometry


@pytest.fixture
def screen():
    return ScreenGeometry(1920, 1080, 531.0, 299.0, 600.0)


def _trace(segments, rate=60.0, jitter_px=0.0, seed=0):
    """Build a gaze trace from (centre_px, n_samples) segments.

    Consecutive segments at different centres produce a single-sample jump
    between them, which is a saccade at any sensible velocity threshold.
    """
    rng = np.random.default_rng(seed)
    points = []
    for centre, n in segments:
        block = np.tile(np.asarray(centre, dtype=float), (n, 1))
        if jitter_px:
            block = block + rng.normal(0.0, jitter_px, size=block.shape)
        points.append(block)
    points = np.vstack(points)
    timestamps = np.arange(len(points)) / rate
    return points, timestamps


def test_velocity_of_a_still_signal_is_zero(screen):
    points, timestamps = _trace([((960.0, 540.0), 30)])
    v = angular_velocity(points, timestamps, screen)
    assert np.all(v == pytest.approx(0.0, abs=1e-9))


def test_velocity_rejects_non_increasing_timestamps(screen):
    points = np.zeros((3, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        angular_velocity(points, np.array([0.0, 0.0, 1.0]), screen)


def test_velocity_first_sample_is_zero(screen):
    points, timestamps = _trace([((100.0, 100.0), 1), ((900.0, 900.0), 1)])
    v = angular_velocity(points, timestamps, screen)
    assert v[0] == 0.0
    assert v[1] > 0.0


def test_single_fixation_is_found(screen):
    points, timestamps = _trace([((960.0, 540.0), 60)])
    fixations = detect_fixations(points, timestamps, screen)

    assert len(fixations) == 1
    assert fixations[0].duration == pytest.approx(59 / 60.0)
    assert fixations[0].centroid_px == pytest.approx([960.0, 540.0])


def test_two_fixations_separated_by_a_saccade(screen):
    points, timestamps = _trace([((400.0, 300.0), 40), ((1400.0, 800.0), 40)])
    fixations = detect_fixations(points, timestamps, screen)

    assert len(fixations) == 2
    assert fixations[0].centroid_px == pytest.approx([400.0, 300.0])
    assert fixations[1].centroid_px == pytest.approx([1400.0, 800.0])


def test_short_fixations_are_discarded(screen):
    """Three samples at 60 Hz is 50 ms, below the 100 ms default."""
    points, timestamps = _trace(
        [((300.0, 300.0), 30), ((900.0, 300.0), 3), ((1500.0, 300.0), 30)]
    )
    fixations = detect_fixations(points, timestamps, screen, min_duration_s=0.100)

    assert len(fixations) == 2
    centroids = [f.centroid_px[0] for f in fixations]
    assert not any(abs(c - 900.0) < 50.0 for c in centroids)


def test_a_single_bad_sample_does_not_split_a_fixation(screen):
    """The reason the merge step exists.

    One stray sample mid-fixation creates two candidates. Without merging,
    both can fall below the duration threshold and the real fixation
    disappears entirely.
    """
    points, timestamps = _trace(
        [((960.0, 540.0), 8), ((1500.0, 540.0), 1), ((960.0, 540.0), 8)]
    )

    merged = detect_fixations(
        points, timestamps, screen, min_duration_s=0.200, merge_max_gap_s=0.075
    )
    unmerged = detect_fixations(
        points, timestamps, screen, min_duration_s=0.200, merge_max_gap_s=0.0
    )

    assert len(merged) == 1
    assert len(unmerged) == 0


def test_merge_does_not_join_distant_clusters(screen):
    """A brief gap is not enough; the centroids must also be close."""
    points, timestamps = _trace([((300.0, 300.0), 20), ((1600.0, 900.0), 20)])
    fixations = detect_fixations(
        points, timestamps, screen, merge_max_gap_s=1.0, merge_max_angle_deg=0.5
    )
    assert len(fixations) == 2


def test_invalid_samples_break_a_fixation(screen):
    """A dropout must not be interpolated across.

    Samples either side of a blink may sit in the same place, but the eye's
    position during the blink is unknown, so merging them invents a fixation
    that was never observed.
    """
    points, timestamps = _trace([((960.0, 540.0), 40)])
    valid = np.ones(40, dtype=bool)
    valid[18:22] = False

    fixations = detect_fixations(points, timestamps, screen, valid=valid, min_duration_s=0.100)

    assert len(fixations) == 2


def test_jittery_but_stationary_signal_is_still_one_fixation(screen):
    """Realistic noise must not shatter a fixation into fragments."""
    points, timestamps = _trace([((960.0, 540.0), 90)], jitter_px=3.0, seed=7)
    fixations = detect_fixations(points, timestamps, screen, velocity_threshold_deg_s=30.0)

    assert len(fixations) == 1
    assert fixations[0].n_samples > 80


def test_empty_input_returns_no_fixations(screen):
    assert detect_fixations(np.zeros((0, 2)), np.zeros(0), screen) == []


def test_valid_mask_length_is_checked(screen):
    points, timestamps = _trace([((960.0, 540.0), 10)])
    with pytest.raises(ValueError, match="same length"):
        detect_fixations(points, timestamps, screen, valid=np.ones(5, dtype=bool))


def test_higher_threshold_merges_more(screen):
    """Sanity check on the knob: a very high threshold classifies everything
    as fixation, so the whole trace collapses to one."""
    points, timestamps = _trace([((300.0, 300.0), 20), ((1600.0, 900.0), 20)])
    fixations = detect_fixations(
        points,
        timestamps,
        screen,
        velocity_threshold_deg_s=100000.0,
        merge_max_angle_deg=1e9,
        merge_max_gap_s=1e9,
    )
    assert len(fixations) == 1
