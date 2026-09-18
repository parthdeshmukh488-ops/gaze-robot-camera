"""Accuracy and precision, reported the way the eye-tracking literature does.

The two numbers answer different questions and a tracker can be good at one and
bad at the other:

- **Accuracy** — how far the reported gaze sits from where the person actually
  looked, on average. A systematic offset. Fixable by calibration.
- **Precision** — how much the reported gaze moves while the person holds
  still. Noise. Not fixable by calibration, only by filtering, and filtering
  costs latency.

Both are in degrees of visual angle. Reporting either in pixels is
uninterpretable without the screen size and viewing distance, which is why
every function here takes a :class:`ScreenGeometry`.

Precision has two standard definitions and they are not interchangeable, so
both are provided explicitly rather than one being called "the" precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import ScreenGeometry


@dataclass(frozen=True)
class AccuracyReport:
    """Angular offset between reported gaze and the true target."""

    mean_deg: float
    median_deg: float
    p95_deg: float
    std_deg: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"accuracy {self.mean_deg:.2f} deg mean, {self.median_deg:.2f} median, "
            f"{self.p95_deg:.2f} p95 (n={self.n_samples})"
        )


def accuracy(
    predicted_px: np.ndarray,
    target_px: np.ndarray,
    geometry: ScreenGeometry,
) -> AccuracyReport:
    """Angular error of each prediction against its target.

    The median and the 95th percentile are reported alongside the mean on
    purpose. Gaze error distributions have a long right tail — a handful of
    blinks or bad frames pull the mean well above what the tracker does
    typically — so a mean on its own flatters or damns a tracker depending on
    how the outliers fell.
    """
    predicted_px = np.atleast_2d(np.asarray(predicted_px, dtype=float))
    target_px = np.atleast_2d(np.asarray(target_px, dtype=float))

    if predicted_px.shape != target_px.shape:
        raise ValueError("predicted_px and target_px must have the same shape")

    finite = np.all(np.isfinite(predicted_px), axis=1) & np.all(np.isfinite(target_px), axis=1)
    if not np.any(finite):
        raise ValueError("no finite prediction/target pairs")

    errors = geometry.angular_distance_deg(predicted_px[finite], target_px[finite])

    return AccuracyReport(
        mean_deg=float(np.mean(errors)),
        median_deg=float(np.median(errors)),
        p95_deg=float(np.percentile(errors, 95)),
        std_deg=float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
        n_samples=int(errors.size),
    )


def precision_rms_s2s(points_px: np.ndarray, geometry: ScreenGeometry) -> float:
    """Sample-to-sample RMS precision, in degrees.

    The root mean square of the angular distance between consecutive samples.
    This is the number that predicts whether a cursor will look like it is
    shivering, because it is sensitive to exactly the high-frequency component
    a viewer perceives as shake.
    """
    points_px = np.atleast_2d(np.asarray(points_px, dtype=float))
    if points_px.shape[0] < 2:
        raise ValueError("need at least two samples for sample-to-sample precision")

    steps = geometry.angular_distance_deg(points_px[:-1], points_px[1:])
    return float(np.sqrt(np.mean(steps**2)))


def precision_sd(points_px: np.ndarray, geometry: ScreenGeometry) -> float:
    """Dispersion precision: SD of the samples about their own centroid, in degrees.

    Lower than the sample-to-sample figure for the same data whenever the noise
    is temporally correlated, which it usually is. Quote which one you used —
    comparing an SD precision against someone else's sample-to-sample precision
    makes a tracker look roughly twice as good as it is.
    """
    points_px = np.atleast_2d(np.asarray(points_px, dtype=float))
    if points_px.shape[0] < 2:
        raise ValueError("need at least two samples for dispersion precision")

    centroid = points_px.mean(axis=0, keepdims=True)
    offsets = geometry.angular_distance_deg(
        np.repeat(centroid, points_px.shape[0], axis=0), points_px
    )
    return float(np.sqrt(np.mean(offsets**2)))


def data_loss_rate(valid: np.ndarray) -> float:
    """Fraction of samples where tracking was lost, in [0, 1].

    Always report this next to accuracy and precision. A tracker that discards
    its hard frames and scores the rest will post excellent numbers on the 60%
    of samples it kept, and the comparison against one that reported everything
    is meaningless.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0:
        raise ValueError("valid must be non-empty")
    return float(1.0 - valid.mean())
