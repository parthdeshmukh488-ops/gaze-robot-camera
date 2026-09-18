"""Dwell selection: look at a target long enough and it commits.

This is the classic gaze-only technique and it carries the classic problem with
it. The eye is a perceptual organ, not a pointing device — you look at things
to read them, not to choose them — so a system that treats looking as choosing
fires selections the user never intended. Jacob named that the Midas touch
problem in 1990 and no dwell implementation escapes it; it can only be traded
against speed via the dwell time.

Two details here matter more than the threshold itself:

**Grace period.** A raw "did the gaze leave the target" test resets the dwell
on every stray sample, and with a webcam-grade signal that means long dwells
essentially never complete. The gaze has to be outside the target continuously
for ``grace_s`` before progress is lost.

**Refractory period.** After a selection, the gaze is usually still sitting on
the target that was just chosen. Without a lockout the same target fires again
immediately, repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import SelectionEvent, SelectionResult, Target, target_at


@dataclass
class DwellSelector:
    """Dwell-time selection with grace and refractory periods.

    Args:
        targets: the selectable regions.
        dwell_s: how long the gaze must rest on a target to commit it. 600 ms
            up to about 1000 ms is the usual range; below roughly 400 ms false
            selections dominate.
        grace_s: how long the gaze may stray off-target without losing
            progress.
        refractory_s: lockout after a selection, during which no target can
            fire.
    """

    targets: list[Target]
    dwell_s: float = 0.800
    grace_s: float = 0.150
    refractory_s: float = 0.500

    _active_id: str | None = field(default=None, repr=False)
    _accumulated: float = field(default=0.0, repr=False)
    _last_on_target_t: float | None = field(default=None, repr=False)
    _last_update_t: float | None = field(default=None, repr=False)
    _counting: bool = field(default=False, repr=False)
    _refractory_until: float = field(default=-np.inf, repr=False)

    def __post_init__(self) -> None:
        if self.dwell_s <= 0:
            raise ValueError("dwell_s must be positive")
        if self.grace_s < 0:
            raise ValueError("grace_s must be non-negative")
        if self.refractory_s < 0:
            raise ValueError("refractory_s must be non-negative")

    def reset(self) -> None:
        self._active_id = None
        self._accumulated = 0.0
        self._last_on_target_t = None
        self._last_update_t = None
        self._counting = False
        self._refractory_until = -np.inf

    def _abandon(self) -> None:
        self._active_id = None
        self._accumulated = 0.0
        self._last_on_target_t = None
        self._counting = False

    def update(self, point_px: np.ndarray, t: float, *, valid: bool = True) -> SelectionResult:
        previous_t = self._last_update_t
        was_counting = self._counting
        self._last_update_t = t
        self._counting = False

        # A lost sample is not evidence that the gaze left the target, so it
        # neither advances nor resets the dwell — it is treated exactly like a
        # sample that strayed, and the grace period decides.
        if not valid:
            return self._handle_off_target(t, event_if_lost=SelectionEvent.NONE)

        hit = target_at(self.targets, point_px)

        if hit is None:
            return self._handle_off_target(t)

        if t < self._refractory_until:
            return SelectionResult(event=SelectionEvent.NONE, target_id=hit.id)

        if hit.id != self._active_id:
            # Moved to a different target: the previous dwell is abandoned
            # outright rather than carried over.
            self._active_id = hit.id
            self._accumulated = 0.0
            self._last_on_target_t = t
            self._counting = True
            return SelectionResult(event=SelectionEvent.ENTERED, target_id=hit.id, progress=0.0)

        # Credit only the interval between two consecutive samples that were
        # both valid and on this target. Measuring elapsed time from when the
        # dwell started instead would hand the user credit for time spent
        # blinking or looking elsewhere, and a long enough dropout would
        # complete a dwell that never happened.
        if was_counting and previous_t is not None:
            self._accumulated += max(t - previous_t, 0.0)
        self._last_on_target_t = t
        self._counting = True

        if self._accumulated >= self.dwell_s:
            selected = self._active_id
            self._abandon()
            self._refractory_until = t + self.refractory_s
            return SelectionResult(
                event=SelectionEvent.SELECTED,
                target_id=selected,
                progress=1.0,
                metadata={"dwell_s": self.dwell_s},
            )

        return SelectionResult(
            event=SelectionEvent.NONE,
            target_id=self._active_id,
            progress=min(1.0, self._accumulated / self.dwell_s),
        )

    def _handle_off_target(
        self, t: float, event_if_lost: SelectionEvent = SelectionEvent.NONE
    ) -> SelectionResult:
        if self._active_id is None:
            return SelectionResult(event=event_if_lost)

        off_for = t - (self._last_on_target_t if self._last_on_target_t is not None else t)
        if off_for > self.grace_s:
            left = self._active_id
            self._abandon()
            return SelectionResult(event=SelectionEvent.LEFT, target_id=left)

        # Inside the grace window: hold progress where it is. It does not
        # advance while the gaze is away, which is the honest choice — the user
        # is not looking at the target, so the dwell should not be earning
        # credit.
        return SelectionResult(
            event=SelectionEvent.NONE,
            target_id=self._active_id,
            progress=min(1.0, self._accumulated / self.dwell_s),
        )
