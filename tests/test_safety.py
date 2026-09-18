"""Tests for the safety layer between the gaze signal and the robot.

This is the part of the project where a bug is not a usability problem, so the
tests are written as the properties that must hold for *any* input, including
adversarial ones: a gaze signal that jumps across the screen, one that stops
arriving, and one that asks for a pose outside the workspace.
"""

import numpy as np
import pytest

from gazectl.robot import SafetyFilter, WorkspaceLimits, gaze_to_pan_tilt

LIMITS = WorkspaceLimits(pan_min=-1.0, pan_max=1.0, tilt_min=-0.5, tilt_max=0.5)


def test_limits_reject_inverted_ranges():
    with pytest.raises(ValueError, match="pan_min"):
        WorkspaceLimits(1.0, -1.0, -0.5, 0.5)
    with pytest.raises(ValueError, match="tilt_min"):
        WorkspaceLimits(-1.0, 1.0, 0.5, -0.5)


def test_clamp_reports_whether_it_acted():
    inside, changed = LIMITS.clamp(np.array([0.5, 0.2]))
    assert not changed
    assert inside == pytest.approx([0.5, 0.2])

    outside, changed = LIMITS.clamp(np.array([5.0, -3.0]))
    assert changed
    assert outside == pytest.approx([1.0, -0.5])


def test_filter_rejects_bad_parameters():
    with pytest.raises(ValueError, match="max_speed_rad_s"):
        SafetyFilter(LIMITS, max_speed_rad_s=0.0)
    with pytest.raises(ValueError, match="tracking_timeout_s"):
        SafetyFilter(LIMITS, tracking_timeout_s=0.0)


def test_starts_at_the_workspace_centre():
    f = SafetyFilter(LIMITS)
    assert f.command == pytest.approx([0.0, 0.0])


def test_command_never_leaves_the_workspace():
    """The property that must hold whatever the tracker does.

    A gaze signal jumping randomly far outside the reachable range must never
    produce a command outside it, on any cycle.
    """
    rng = np.random.default_rng(0)
    f = SafetyFilter(LIMITS, max_speed_rad_s=100.0)

    for i in range(500):
        wild = rng.uniform(-50.0, 50.0, size=2)
        command = f.step(wild, t=i / 60.0)

        assert LIMITS.pan_min <= command[0] <= LIMITS.pan_max
        assert LIMITS.tilt_min <= command[1] <= LIMITS.tilt_max


def test_speed_limit_is_respected():
    """A step input must become a ramp, not a lurch."""
    max_speed = 0.5
    rate = 60.0
    f = SafetyFilter(LIMITS, max_speed_rad_s=max_speed)

    previous = f.command
    for i in range(1, 120):
        t = i / rate
        command = f.step(np.array([1.0, 0.5]), t=t)
        step_size = np.linalg.norm(command - previous)

        assert step_size <= max_speed / rate + 1e-9
        previous = command


def test_reaches_the_target_eventually():
    f = SafetyFilter(LIMITS, max_speed_rad_s=2.0)
    for i in range(600):
        command = f.step(np.array([0.8, 0.3]), t=i / 60.0)
    assert command == pytest.approx([0.8, 0.3], abs=1e-6)


def test_holds_when_tracking_is_lost():
    f = SafetyFilter(LIMITS, max_speed_rad_s=2.0, tracking_timeout_s=0.2)

    for i in range(60):
        f.step(np.array([0.5, 0.2]), t=i / 60.0)
    before = f.command
    assert not f.holding

    for i in range(60, 120):
        command = f.step(None, t=i / 60.0, valid=False)

    assert f.holding
    assert command == pytest.approx(before)


def test_brief_dropout_does_not_trigger_the_hold():
    """Single dropped frames are routine and must not stop the robot."""
    f = SafetyFilter(LIMITS, max_speed_rad_s=2.0, tracking_timeout_s=0.3)

    for i in range(30):
        f.step(np.array([0.5, 0.2]), t=i / 60.0)

    f.step(None, t=30 / 60.0, valid=False)
    f.step(None, t=31 / 60.0, valid=False)

    assert not f.holding


def test_recovers_after_tracking_returns():
    f = SafetyFilter(LIMITS, max_speed_rad_s=2.0, tracking_timeout_s=0.2)

    for i in range(60):
        f.step(None, t=i / 60.0, valid=False)
    assert f.holding

    f.step(np.array([0.5, 0.2]), t=1.0)
    assert not f.holding


def test_dropout_time_cannot_be_spent_as_one_large_step():
    """The subtle one.

    If dt were only measured across valid cycles, a two-second dropout
    followed by one good sample would authorise a two-second-worth of travel
    in a single cycle — a lurch precisely when the system has just recovered
    from not knowing where the user was looking.
    """
    rate = 60.0
    max_speed = 0.5
    f = SafetyFilter(LIMITS, max_speed_rad_s=max_speed, tracking_timeout_s=10.0)

    f.step(np.array([0.0, 0.0]), t=0.0)
    for i in range(1, 120):
        f.step(None, t=i / rate, valid=False)

    before = f.command
    after = f.step(np.array([1.0, 0.5]), t=120 / rate)

    assert np.linalg.norm(after - before) <= max_speed / rate + 1e-9


def test_reset_returns_to_home():
    f = SafetyFilter(LIMITS, max_speed_rad_s=10.0)
    for i in range(120):
        f.step(np.array([0.9, 0.4]), t=i / 60.0)

    f.reset()
    assert f.command == pytest.approx([0.0, 0.0])
    assert not f.holding


def test_custom_home_is_honoured():
    f = SafetyFilter(LIMITS, home=np.array([0.3, -0.2]))
    assert f.command == pytest.approx([0.3, -0.2])


def test_gaze_to_pan_tilt_centre_is_neutral():
    command = gaze_to_pan_tilt(
        np.array([960.0, 540.0]), 1920, 1080, pan_range_rad=2.0, tilt_range_rad=1.0
    )
    assert command == pytest.approx([0.0, 0.0])


def test_gaze_to_pan_tilt_sign_convention():
    """Right of centre pans positive; below centre tilts negative.

    The vertical flip is the one to get wrong: screen y grows downwards while
    tilt grows upwards.
    """
    right = gaze_to_pan_tilt(
        np.array([1920.0, 540.0]), 1920, 1080, pan_range_rad=2.0, tilt_range_rad=1.0
    )
    assert right[0] > 0

    low = gaze_to_pan_tilt(
        np.array([960.0, 1080.0]), 1920, 1080, pan_range_rad=2.0, tilt_range_rad=1.0
    )
    assert low[1] < 0


def test_gaze_to_pan_tilt_spans_the_requested_range():
    left = gaze_to_pan_tilt(
        np.array([0.0, 540.0]), 1920, 1080, pan_range_rad=2.0, tilt_range_rad=1.0
    )
    right = gaze_to_pan_tilt(
        np.array([1920.0, 540.0]), 1920, 1080, pan_range_rad=2.0, tilt_range_rad=1.0
    )
    assert (right[0] - left[0]) == pytest.approx(2.0)
