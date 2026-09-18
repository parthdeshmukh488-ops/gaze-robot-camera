"""Look-and-confirm selection: the eye points, a button commits.

The eye chooses the target and an explicit action — a key, a footswitch, a
button — commits it. This gives up the hands-free property that makes dwell
attractive and buys back the thing dwell cannot have: the user can look at
something without selecting it.

That makes it the honest middle condition in the study. Dwell is fast and
fires when you did not mean it; the mouse never fires when you did not mean it
and is slow; look-and-confirm is meant to sit between them, and the study
exists to find out whether it actually does.

The implementation detail that decides whether it works is the **confirm
window**. A user who presses the key at the moment they look away would
otherwise commit the wrong target, or nothing. Holding the last target for a
short window after the gaze leaves absorbs that, because the motor act of
pressing a key reliably lags the decision that was made while looking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import SelectionEvent, SelectionResult, Target, target_at


@dataclass
class LookAndConfirmSelector:
    """Gaze selects a candidate; an external confirm signal commits it.

    Args:
        targets: the selectable regions.
        confirm_window_s: how long a candidate stays confirmable after the gaze
            leaves it. 300 ms covers a normal key press that starts as the eye
            moves on.
        min_fixation_s: how long the gaze must rest on a target before it can
            become a candidate at all. Stops targets that were merely crossed
            en route from being confirmable.
    """

    targets: list[Target]
    confirm_window_s: float = 0.300
    min_fixation_s: float = 0.100

    _candidate_id: str | None = field(default=None, repr=False)
    _candidate_since: float | None = field(default=None, repr=False)
    _candidate_expires: float = field(default=-np.inf, repr=False)
    _qualified: bool = field(default=False, repr=False)
    _last_t: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.confirm_window_s < 0:
            raise ValueError("confirm_window_s must be non-negative")
        if self.min_fixation_s < 0:
            raise ValueError("min_fixation_s must be non-negative")

    def reset(self) -> None:
        self._candidate_id = None
        self._candidate_since = None
        self._candidate_expires = -np.inf
        self._qualified = False
        self._last_t = 0.0

    @property
    def candidate(self) -> str | None:
        """The target a confirm would currently commit, if any."""
        if self._candidate_id is None or not self._qualified:
            return None
        if self._last_t > self._candidate_expires:
            return None
        return self._candidate_id

    def update(self, point_px: np.ndarray, t: float, *, valid: bool = True) -> SelectionResult:
        self._last_t = t

        if not valid:
            # Tracking loss does not clear the candidate; the confirm window is
            # what expires it. Clearing here would make the technique fail
            # exactly during the blink that often accompanies a key press.
            return SelectionResult(event=SelectionEvent.NONE, target_id=self.candidate)

        hit = target_at(self.targets, point_px)

        if hit is not None:
            if hit.id != self._candidate_id:
                self._candidate_id = hit.id
                self._candidate_since = t
                self._qualified = False
                self._candidate_expires = np.inf
                return SelectionResult(event=SelectionEvent.ENTERED, target_id=hit.id)

            self._candidate_expires = np.inf
            if not self._qualified:
                dwelt = t - (self._candidate_since if self._candidate_since is not None else t)
                if dwelt >= self.min_fixation_s:
                    self._qualified = True

            return SelectionResult(
                event=SelectionEvent.NONE,
                target_id=hit.id,
                progress=1.0 if self._qualified else 0.0,
            )

        # Off every target: start the confirm window if one is not running.
        if self._candidate_id is not None and np.isinf(self._candidate_expires):
            self._candidate_expires = t + self.confirm_window_s
            return SelectionResult(event=SelectionEvent.LEFT, target_id=self._candidate_id)

        if self._candidate_id is not None and t > self._candidate_expires:
            self._candidate_id = None
            self._candidate_since = None
            self._qualified = False

        return SelectionResult(event=SelectionEvent.NONE, target_id=self.candidate)

    def confirm(self, t: float) -> SelectionResult:
        """Commit the current candidate. Call this on the confirm input.

        Returns a SELECTED result if a candidate was live, CANCELLED otherwise
        — a confirm with nothing under it is recorded rather than dropped,
        because in the study those presses are the error rate.
        """
        self._last_t = t
        candidate = self.candidate

        if candidate is None:
            return SelectionResult(event=SelectionEvent.CANCELLED)

        self._candidate_id = None
        self._candidate_since = None
        self._qualified = False
        self._candidate_expires = -np.inf

        return SelectionResult(event=SelectionEvent.SELECTED, target_id=candidate, progress=1.0)
