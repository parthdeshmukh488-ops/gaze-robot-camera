"""The safety layer between a gaze signal and a moving robot.

This module exists because of one fact: a gaze signal fails *silently*. A blink
gives you nothing, a bad frame gives you a plausible-looking point in the wrong
place, and the person looking away gives you a perfectly valid reading of
somewhere they have no intention of pointing a camera. None of those announce
themselves. If the command path trusts the tracker, the robot swings at the
first bad frame.

So the tracker is not trusted. Everything it produces passes through three
independent gates, and each one is sufficient on its own:

1. **Workspace clamp** — the commanded pan/tilt is confined to the reachable
   range. This one is a hard geometric fact and is applied last, so nothing
   downstream can undo it.
2. **Rate limit** — the command may not change faster than a set angular
   speed, so a jump in the gaze signal becomes a ramp rather than a lurch.
3. **Watchdog** — if no valid sample arrives within a timeout, the robot is
   commanded to hold. Failure is silent, so absence of data has to be treated
   as a stop condition rather than as "keep doing the last thing".

The ordering is deliberate: rate limiting before clamping would let the clamp
introduce a step that the rate limiter never sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WorkspaceLimits:
    """Reachable pan/tilt range of the camera head, in radians."""

    pan_min: float
    pan_max: float
    tilt_min: float
    tilt_max: float

    def __post_init__(self) -> None:
        if self.pan_min >= self.pan_max:
            raise ValueError("pan_min must be below pan_max")
        if self.tilt_min >= self.tilt_max:
            raise ValueError("tilt_min must be below tilt_max")

    def clamp(self, pan_tilt: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp a command into the workspace.

        Returns:
            The clamped command, and whether clamping changed anything.
        """
        pan_tilt = np.asarray(pan_tilt, dtype=float)
        clamped = np.array(
            [
                np.clip(pan_tilt[0], self.pan_min, self.pan_max),
                np.clip(pan_tilt[1], self.tilt_min, self.tilt_max),
            ]
        )
        return clamped, not np.allclose(clamped, pan_tilt)


@dataclass
class SafetyFilter:
    """Rate-limited, workspace-clamped, watchdogged command path.

    Args:
        limits: reachable pan/tilt range.
        max_speed_rad_s: largest permitted change in commanded angle per
            second, per axis.
        tracking_timeout_s: how long without a valid sample before the filter
            holds position.
        home: command to fall back to on :meth:`reset`. Defaults to the
            workspace centre.
    """

    limits: WorkspaceLimits
    max_speed_rad_s: float = 1.0
    tracking_timeout_s: float = 0.300
    home: np.ndarray | None = None

    _command: np.ndarray | None = field(default=None, repr=False)
    _last_valid_t: float | None = field(default=None, repr=False)
    _last_step_t: float | None = field(default=None, repr=False)
    _holding: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_speed_rad_s <= 0:
            raise ValueError("max_speed_rad_s must be positive")
        if self.tracking_timeout_s <= 0:
            raise ValueError("tracking_timeout_s must be positive")
        if self.home is None:
            self.home = np.array(
                [
                    (self.limits.pan_min + self.limits.pan_max) / 2.0,
                    (self.limits.tilt_min + self.limits.tilt_max) / 2.0,
                ]
            )
        self.reset()

    def reset(self) -> None:
        self._command = np.asarray(self.home, dtype=float).copy()
        self._last_valid_t = None
        self._last_step_t = None
        self._holding = False

    @property
    def command(self) -> np.ndarray:
        """The most recent command actually issued to the robot."""
        return self._command.copy()

    @property
    def holding(self) -> bool:
        """True when the watchdog has stopped following the gaze."""
        return self._holding

    def step(
        self,
        desired_pan_tilt: np.ndarray | None,
        t: float,
        *,
        valid: bool = True,
    ) -> np.ndarray:
        """Advance the command by one control cycle.

        Args:
            desired_pan_tilt: what the gaze mapping asks for, or None when
                there is nothing to ask.
            t: timestamp in seconds.
            valid: False when the tracker lost the eye on this cycle.

        Returns:
            The command to send to the robot.
        """
        if self._last_valid_t is None:
            self._last_valid_t = t

        # dt is measured across every cycle, valid or not, so that a burst of
        # dropped frames followed by a good one cannot be spent as a single
        # large rate-limited step.
        dt = max(t - self._last_step_t, 0.0) if self._last_step_t is not None else 0.0
        self._last_step_t = t

        usable = valid and desired_pan_tilt is not None

        if usable:
            self._last_valid_t = t
            self._holding = False
        else:
            since_valid = t - self._last_valid_t
            if since_valid >= self.tracking_timeout_s:
                # Hold. Not return-to-home: a robot that sweeps back to centre
                # every time the user blinks is worse than one that stops, and
                # the sweep itself is unexpected motion.
                self._holding = True
            return self.command

        target, _ = self.limits.clamp(np.asarray(desired_pan_tilt, dtype=float))

        max_delta = self.max_speed_rad_s * dt
        delta = target - self._command
        distance = float(np.linalg.norm(delta))

        if distance > max_delta and distance > 0.0:
            delta = delta * (max_delta / distance)

        self._command, _ = self.limits.clamp(self._command + delta)
        return self.command


def gaze_to_pan_tilt(
    point_px: np.ndarray,
    screen_width_px: int,
    screen_height_px: int,
    *,
    pan_range_rad: float,
    tilt_range_rad: float,
) -> np.ndarray:
    """Map a screen gaze point to a pan/tilt command.

    A deliberately simple linear map from the normalised screen position to an
    angular range, centred so that looking at the middle of the screen means
    the camera's neutral pose. The nonlinearity of the real eye-to-screen
    relationship has already been absorbed by the per-user calibration fit, so
    adding another one here would be fitting the same effect twice.
    """
    point_px = np.asarray(point_px, dtype=float)
    u = point_px[0] / screen_width_px - 0.5
    v = point_px[1] / screen_height_px - 0.5
    return np.array([u * pan_range_rad, -v * tilt_range_rad])
