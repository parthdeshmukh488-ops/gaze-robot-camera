"""Tests for the two gaze selection techniques.

Both are state machines, so the tests are written as scripted sample streams
where the correct sequence of events is known. The tests that matter most are
the ones covering the failure modes the techniques exist to manage: the Midas
touch for dwell, and the timing of the confirm press for look-and-confirm.
"""

import numpy as np
import pytest

from gazectl.selection import (
    DwellSelector,
    LookAndConfirmSelector,
    SelectionEvent,
    Target,
    target_at,
)

TARGETS = [
    Target(id="a", x=100.0, y=100.0, width=200.0, height=200.0),
    Target(id="b", x=600.0, y=100.0, width=200.0, height=200.0),
]

ON_A = np.array([200.0, 200.0])
ON_B = np.array([700.0, 200.0])
OFF = np.array([1500.0, 900.0])


def _feed(selector, point, start, duration, rate=60.0):
    """Feed a steady stream of one point, returning every result."""
    out = []
    n = int(duration * rate)
    for i in range(n):
        out.append(selector.update(point, start + i / rate))
    return out


def test_target_contains_and_centre():
    t = TARGETS[0]
    assert t.contains(np.array([100.0, 100.0]))  # corner is inside
    assert t.contains(np.array([300.0, 300.0]))
    assert not t.contains(np.array([99.0, 200.0]))
    assert t.centre_px == pytest.approx([200.0, 200.0])


def test_target_at_returns_none_off_target():
    assert target_at(TARGETS, OFF) is None
    assert target_at(TARGETS, ON_B).id == "b"


def test_dwell_rejects_bad_parameters():
    with pytest.raises(ValueError, match="dwell_s"):
        DwellSelector(TARGETS, dwell_s=0.0)
    with pytest.raises(ValueError, match="grace_s"):
        DwellSelector(TARGETS, grace_s=-1.0)


def test_dwell_selects_after_the_dwell_time():
    sel = DwellSelector(TARGETS, dwell_s=0.5, refractory_s=0.0)
    results = _feed(sel, ON_A, 0.0, 0.7)

    selections = [r for r in results if r.event is SelectionEvent.SELECTED]
    assert len(selections) == 1
    assert selections[0].target_id == "a"


def test_dwell_does_not_select_too_early():
    sel = DwellSelector(TARGETS, dwell_s=0.8)
    results = _feed(sel, ON_A, 0.0, 0.5)

    assert not any(r.event is SelectionEvent.SELECTED for r in results)
    assert results[-1].progress == pytest.approx(0.5 / 0.8, abs=0.05)


def test_dwell_emits_entered_on_arrival():
    sel = DwellSelector(TARGETS, dwell_s=1.0)
    first = sel.update(ON_A, 0.0)
    assert first.event is SelectionEvent.ENTERED
    assert first.target_id == "a"


def test_dwell_resets_when_gaze_leaves_for_longer_than_grace():
    sel = DwellSelector(TARGETS, dwell_s=0.5, grace_s=0.1)

    _feed(sel, ON_A, 0.0, 0.3)
    away = _feed(sel, OFF, 0.3, 0.3)
    assert any(r.event is SelectionEvent.LEFT for r in away)

    back = _feed(sel, ON_A, 0.6, 0.3)
    assert not any(r.event is SelectionEvent.SELECTED for r in back)


def test_dwell_survives_a_brief_stray_within_grace():
    """A raw in/out test would reset here and long dwells would never finish."""
    sel = DwellSelector(TARGETS, dwell_s=0.5, grace_s=0.15, refractory_s=0.0)

    _feed(sel, ON_A, 0.0, 0.3)
    _feed(sel, OFF, 0.3, 0.05)  # brief stray, inside grace
    back = _feed(sel, ON_A, 0.35, 0.3)

    assert any(r.event is SelectionEvent.SELECTED for r in back)


def test_dwell_switching_target_abandons_progress():
    sel = DwellSelector(TARGETS, dwell_s=0.5, refractory_s=0.0)

    _feed(sel, ON_A, 0.0, 0.4)
    on_b = _feed(sel, ON_B, 0.4, 0.3)

    assert on_b[0].event is SelectionEvent.ENTERED
    assert not any(r.event is SelectionEvent.SELECTED for r in on_b)


def test_dwell_refractory_prevents_immediate_repeat():
    """Without this, staying on a target fires it over and over."""
    sel = DwellSelector(TARGETS, dwell_s=0.3, refractory_s=1.0)
    results = _feed(sel, ON_A, 0.0, 1.2)

    selections = [r for r in results if r.event is SelectionEvent.SELECTED]
    assert len(selections) == 1


def test_dwell_without_refractory_repeats():
    """The Midas touch, demonstrated rather than asserted.

    With the lockout removed, simply continuing to look fires the same target
    again and again. This is the behaviour the refractory period exists to
    suppress, and it is the core usability problem with dwell.
    """
    sel = DwellSelector(TARGETS, dwell_s=0.3, refractory_s=0.0)
    results = _feed(sel, ON_A, 0.0, 1.2)

    selections = [r for r in results if r.event is SelectionEvent.SELECTED]
    assert len(selections) >= 3


def test_dwell_invalid_samples_do_not_advance_progress():
    sel = DwellSelector(TARGETS, dwell_s=0.5, grace_s=10.0, refractory_s=0.0)

    sel.update(ON_A, 0.0)
    for i in range(60):
        sel.update(ON_A, 0.01 + i / 60.0, valid=False)

    # A full second of invalid samples must not have completed the dwell.
    assert sel.update(ON_A, 1.02).event is not SelectionEvent.SELECTED


def test_confirm_requires_a_candidate():
    sel = LookAndConfirmSelector(TARGETS)
    result = sel.confirm(0.0)
    assert result.event is SelectionEvent.CANCELLED


def test_confirm_selects_the_looked_at_target():
    sel = LookAndConfirmSelector(TARGETS, min_fixation_s=0.1)
    _feed(sel, ON_A, 0.0, 0.3)

    result = sel.confirm(0.3)
    assert result.event is SelectionEvent.SELECTED
    assert result.target_id == "a"


def test_confirm_ignores_a_target_merely_crossed():
    """min_fixation_s stops a target passed over en route being confirmable."""
    sel = LookAndConfirmSelector(TARGETS, min_fixation_s=0.2)
    _feed(sel, ON_A, 0.0, 0.05)

    assert sel.confirm(0.05).event is SelectionEvent.CANCELLED


def test_confirm_works_just_after_the_gaze_leaves():
    """The reason the confirm window exists.

    Pressing a key lags the decision that was made while looking, so a press
    landing shortly after the eye moves on must still commit the target that
    was chosen.
    """
    sel = LookAndConfirmSelector(TARGETS, confirm_window_s=0.3, min_fixation_s=0.1)

    _feed(sel, ON_A, 0.0, 0.3)
    _feed(sel, OFF, 0.3, 0.1)  # gaze moved on

    result = sel.confirm(0.4)
    assert result.event is SelectionEvent.SELECTED
    assert result.target_id == "a"


def test_confirm_window_expires():
    sel = LookAndConfirmSelector(TARGETS, confirm_window_s=0.2, min_fixation_s=0.1)

    _feed(sel, ON_A, 0.0, 0.3)
    _feed(sel, OFF, 0.3, 0.5)  # well past the window

    assert sel.confirm(0.8).event is SelectionEvent.CANCELLED


def test_confirm_never_fires_without_a_press():
    """The property that distinguishes it from dwell.

    However long the gaze rests on a target, nothing is selected until the
    confirm input arrives. This is what makes looking safe.
    """
    sel = LookAndConfirmSelector(TARGETS, min_fixation_s=0.1)
    results = _feed(sel, ON_A, 0.0, 5.0)

    assert not any(r.event is SelectionEvent.SELECTED for r in results)


def test_confirm_survives_a_blink():
    """Tracking loss must not clear the candidate.

    A blink often coincides with the key press, so dropping the candidate on
    an invalid sample would make the technique fail exactly when it is used.
    """
    sel = LookAndConfirmSelector(TARGETS, min_fixation_s=0.1)
    _feed(sel, ON_A, 0.0, 0.3)

    for i in range(6):
        sel.update(ON_A, 0.3 + i / 60.0, valid=False)

    assert sel.confirm(0.4).event is SelectionEvent.SELECTED


def test_confirm_switches_candidate_with_gaze():
    sel = LookAndConfirmSelector(TARGETS, min_fixation_s=0.1)
    _feed(sel, ON_A, 0.0, 0.3)
    _feed(sel, ON_B, 0.3, 0.3)

    assert sel.confirm(0.6).target_id == "b"


def test_reset_clears_both_techniques():
    dwell = DwellSelector(TARGETS, dwell_s=0.5)
    _feed(dwell, ON_A, 0.0, 0.4)
    dwell.reset()
    assert _feed(dwell, ON_A, 1.0, 0.3)[0].event is SelectionEvent.ENTERED

    confirm = LookAndConfirmSelector(TARGETS, min_fixation_s=0.1)
    _feed(confirm, ON_A, 0.0, 0.3)
    confirm.reset()
    assert confirm.confirm(0.5).event is SelectionEvent.CANCELLED
