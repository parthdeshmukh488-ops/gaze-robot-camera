"""Tests for the pixel-to-visual-angle conversion.

These matter more than they look. Every accuracy figure in the project passes
through this conversion, so an error here silently rescales every result.
"""

import math

import numpy as np
import pytest

from gazectl.geometry import ScreenGeometry


@pytest.fixture
def screen():
    """A 24-inch 1920x1080 monitor at a typical desk distance."""
    return ScreenGeometry(
        width_px=1920,
        height_px=1080,
        width_mm=531.0,
        height_mm=299.0,
        viewing_distance_mm=600.0,
    )


def test_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="width_px"):
        ScreenGeometry(0, 1080, 531.0, 299.0, 600.0)
    with pytest.raises(ValueError, match="viewing_distance_mm"):
        ScreenGeometry(1920, 1080, 531.0, 299.0, -1.0)


def test_zero_angle_between_a_point_and_itself(screen):
    p = np.array([[960.0, 540.0]])
    assert screen.angular_distance_deg(p, p)[0] == pytest.approx(0.0, abs=1e-9)


def test_angle_matches_hand_computation_at_centre(screen):
    """A known horizontal offset from centre, checked against atan by hand."""
    centre = np.array([[960.0, 540.0]])
    offset_px = 100.0
    other = np.array([[960.0 + offset_px, 540.0]])

    offset_mm = offset_px * screen.mm_per_px_x
    expected = math.degrees(math.atan2(offset_mm, screen.viewing_distance_mm))

    assert screen.angular_distance_deg(centre, other)[0] == pytest.approx(expected, rel=1e-9)


def test_angle_is_symmetric(screen):
    a = np.array([[300.0, 200.0]])
    b = np.array([[1500.0, 900.0]])
    assert screen.angular_distance_deg(a, b)[0] == pytest.approx(
        screen.angular_distance_deg(b, a)[0], rel=1e-12
    )


def test_same_pixel_distance_subtends_less_angle_further_from_centre(screen):
    """The whole reason for using the true subtended angle.

    A 100 px step near the screen edge is further from the eye and more
    oblique to it than the same step at the centre, so it subtends a smaller
    angle. The small-angle approximation d/D misses this entirely, and it is
    the difference that matters when scoring gaze accuracy at the edges.
    """
    at_centre = screen.angular_distance_deg(
        np.array([[960.0, 540.0]]), np.array([[1060.0, 540.0]])
    )[0]
    at_edge = screen.angular_distance_deg(
        np.array([[1750.0, 540.0]]), np.array([[1850.0, 540.0]])
    )[0]

    assert at_edge < at_centre


def test_vectorises_over_many_pairs(screen):
    a = np.random.default_rng(0).uniform([0, 0], [1920, 1080], size=(50, 2))
    b = np.random.default_rng(1).uniform([0, 0], [1920, 1080], size=(50, 2))

    angles = screen.angular_distance_deg(a, b)

    assert angles.shape == (50,)
    assert np.all(angles >= 0.0)
    for i in range(50):
        one = screen.angular_distance_deg(a[i : i + 1], b[i : i + 1])[0]
        assert angles[i] == pytest.approx(one, rel=1e-12)


def test_deg_to_px_round_trips_near_centre(screen):
    """One degree converted to pixels and back should return one degree."""
    px = screen.deg_to_px_at_centre(1.0)
    centre = np.array([[960.0, 540.0]])
    shifted = np.array([[960.0 + px, 540.0]])

    assert screen.angular_distance_deg(centre, shifted)[0] == pytest.approx(1.0, rel=1e-9)


def test_larger_viewing_distance_shrinks_the_angle():
    near = ScreenGeometry(1920, 1080, 531.0, 299.0, 400.0)
    far = ScreenGeometry(1920, 1080, 531.0, 299.0, 800.0)

    a = np.array([[960.0, 540.0]])
    b = np.array([[1200.0, 540.0]])

    assert far.angular_distance_deg(a, b)[0] < near.angular_distance_deg(a, b)[0]
