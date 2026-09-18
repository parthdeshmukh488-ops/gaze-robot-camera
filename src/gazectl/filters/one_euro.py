"""The One Euro filter (Casiez, Roussel and Vogel, CHI 2012).

Gaze signals have a problem that a fixed low-pass filter cannot solve: during a
fixation you want heavy smoothing, because the jitter is all noise and it makes
a cursor unusable; during a saccade you want almost none, because the lag shows
up directly as the cursor trailing the eye.

The One Euro filter adapts its cutoff to the observed speed of the signal — low
cutoff when slow, high cutoff when fast — which is exactly that trade-off with
one interpretable knob per side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at a given cutoff."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class _LowPass:
    """First-order low-pass with a settable per-sample alpha."""

    _value: np.ndarray | None = None

    @property
    def initialised(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> np.ndarray | None:
        return self._value

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if self._value is None:
            self._value = np.asarray(x, dtype=float).copy()
        else:
            self._value = alpha * np.asarray(x, dtype=float) + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


@dataclass
class OneEuroFilter:
    """Speed-adaptive low-pass filter for a vector-valued signal.

    Args:
        min_cutoff_hz: cutoff as the signal approaches standstill. Lower means
            steadier during a fixation and slower to react. This is the knob to
            turn if the cursor shivers while the user holds still.
        beta: how sharply the cutoff opens up with speed. Higher means less lag
            during a saccade and more jitter. This is the knob to turn if the
            cursor lags behind fast eye movements.
        d_cutoff_hz: cutoff of the low-pass applied to the speed estimate
            itself. The paper's default of 1.0 is almost always right.

    The two knobs are tuned in that order: raise ``beta`` until lag is
    acceptable, then lower ``min_cutoff_hz`` until jitter is acceptable.
    """

    min_cutoff_hz: float = 1.0
    beta: float = 0.007
    d_cutoff_hz: float = 1.0

    _x: _LowPass = field(default_factory=_LowPass, repr=False)
    _dx: _LowPass = field(default_factory=_LowPass, repr=False)
    _t_prev: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.min_cutoff_hz <= 0:
            raise ValueError("min_cutoff_hz must be positive")
        if self.d_cutoff_hz <= 0:
            raise ValueError("d_cutoff_hz must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        """Filter one sample.

        Args:
            x: the sample, scalar or vector.
            t: its timestamp in seconds. Must be non-decreasing.

        Returns:
            The filtered sample, same shape as ``x``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))

        if self._t_prev is None:
            self._t_prev = t
            self._dx(np.zeros_like(x), 1.0)
            return self._x(x, 1.0)

        dt = t - self._t_prev
        if dt <= 0:
            # Duplicate or out-of-order timestamp. Returning the last estimate
            # is the only safe answer: a zero or negative dt would blow up the
            # alpha computation, and dropped frames are routine in a webcam
            # pipeline.
            return self._x.value if self._x.initialised else x
        self._t_prev = t

        prev = self._x.value
        dx = (x - prev) / dt if prev is not None else np.zeros_like(x)
        dx_hat = self._dx(dx, _alpha(self.d_cutoff_hz, dt))

        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff_hz + self.beta * speed
        return self._x(x, _alpha(cutoff, dt))
