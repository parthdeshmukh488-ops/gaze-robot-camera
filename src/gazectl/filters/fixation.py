"""I-VT fixation detection (Salvucci and Goldberg, ETRA 2000).

Velocity-threshold identification is the simplest fixation classifier that
works, and for a 30 Hz webcam signal it is also close to the most complex one
worth running: dispersion- and model-based classifiers need a sample rate the
hardware does not have.

The pipeline is the standard one, in this order:

1. point-to-point angular velocity, in degrees per second
2. threshold it — below is fixation, above is saccade
3. collapse runs of fixation samples into candidate fixations
4. merge candidates separated by a brief, small-amplitude gap
5. discard whatever is left that is too short

Steps 4 and 5 are not decoration. Without the merge, a single noisy sample in
the middle of a fixation splits it in two and both halves may then fail the
duration test, so one bad sample can delete a real fixation entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import ScreenGeometry


@dataclass(frozen=True)
class Fixation:
    """One detected fixation."""

    start_index: int
    end_index: int  # inclusive
    start_time: float
    end_time: float
    centroid_px: np.ndarray

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_samples(self) -> int:
        return self.end_index - self.start_index + 1


def angular_velocity(
    points_px: np.ndarray,
    timestamps: np.ndarray,
    geometry: ScreenGeometry,
) -> np.ndarray:
    """Point-to-point angular velocity in degrees per second.

    The first sample has no predecessor, so its velocity is reported as 0.0 and
    it is therefore always classified as fixation. That is the conventional
    choice and it is harmless — a real saccade lasts several samples.
    """
    points_px = np.asarray(points_px, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError(f"points_px must be (n, 2), got {points_px.shape}")
    if timestamps.shape[0] != points_px.shape[0]:
        raise ValueError("points_px and timestamps must have the same length")

    n = points_px.shape[0]
    if n < 2:
        return np.zeros(n)

    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")

    step_deg = geometry.angular_distance_deg(points_px[:-1], points_px[1:])
    return np.concatenate([[0.0], step_deg / dt])


def detect_fixations(
    points_px: np.ndarray,
    timestamps: np.ndarray,
    geometry: ScreenGeometry,
    *,
    velocity_threshold_deg_s: float = 30.0,
    min_duration_s: float = 0.100,
    merge_max_gap_s: float = 0.075,
    merge_max_angle_deg: float = 0.5,
    valid: np.ndarray | None = None,
) -> list[Fixation]:
    """Classify a gaze trace into fixations.

    Args:
        points_px: (n, 2) gaze points in screen pixels.
        timestamps: (n,) strictly increasing times in seconds.
        geometry: screen geometry, for the pixel-to-degree conversion.
        velocity_threshold_deg_s: the I-VT threshold. 30 deg/s is the usual
            starting point for remote trackers; raise it for a noisier signal.
        min_duration_s: shortest accepted fixation. 100 ms is the low end of
            what the literature treats as a real fixation.
        merge_max_gap_s: candidates closer in time than this may be merged.
        merge_max_angle_deg: and only if their centroids are this close.
        valid: optional (n,) boolean mask. False marks a sample where tracking
            was lost; those samples never belong to a fixation and they break a
            run, because interpolating across a blink invents a fixation that
            did not happen.

    Returns:
        Fixations in time order.
    """
    points_px = np.asarray(points_px, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n = points_px.shape[0]

    if n == 0:
        return []

    if valid is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape[0] != n:
            raise ValueError("valid must have the same length as points_px")

    velocity = angular_velocity(points_px, timestamps, geometry)
    is_fixation = (velocity < velocity_threshold_deg_s) & valid

    runs = _runs_of_true(is_fixation)
    candidates = [
        Fixation(
            start_index=lo,
            end_index=hi,
            start_time=float(timestamps[lo]),
            end_time=float(timestamps[hi]),
            centroid_px=points_px[lo : hi + 1].mean(axis=0),
        )
        for lo, hi in runs
    ]

    merged = _merge_adjacent(
        candidates,
        points_px,
        timestamps,
        geometry,
        max_gap_s=merge_max_gap_s,
        max_angle_deg=merge_max_angle_deg,
        valid=valid,
    )

    return [f for f in merged if f.duration >= min_duration_s]


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs for each run of True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_adjacent(
    fixations: list[Fixation],
    points_px: np.ndarray,
    timestamps: np.ndarray,
    geometry: ScreenGeometry,
    *,
    max_gap_s: float,
    max_angle_deg: float,
    valid: np.ndarray,
) -> list[Fixation]:
    """Merge candidates split by a brief, small-amplitude interruption."""
    if not fixations:
        return []

    out = [fixations[0]]
    for nxt in fixations[1:]:
        cur = out[-1]
        gap = nxt.start_time - cur.end_time
        separation = float(geometry.angular_distance_deg(cur.centroid_px, nxt.centroid_px)[0])

        # A gap containing a tracking dropout is never merged across: we do not
        # know where the eye went while the signal was missing.
        gap_valid = bool(np.all(valid[cur.end_index + 1 : nxt.start_index]))

        if gap <= max_gap_s and separation <= max_angle_deg and gap_valid:
            lo, hi = cur.start_index, nxt.end_index
            out[-1] = Fixation(
                start_index=lo,
                end_index=hi,
                start_time=float(timestamps[lo]),
                end_time=float(timestamps[hi]),
                centroid_px=points_px[lo : hi + 1].mean(axis=0),
            )
        else:
            out.append(nxt)
    return out
