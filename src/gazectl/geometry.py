"""Screen geometry and the pixel-to-visual-angle conversion.

Every accuracy number in this project is quoted in degrees of visual angle
rather than pixels, because pixels are meaningless without knowing how far away
the observer sat. This module owns that conversion and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScreenGeometry:
    """A physical screen and where the observer sits relative to it.

    Args:
        width_px: horizontal resolution.
        height_px: vertical resolution.
        width_mm: physical width of the active area.
        height_mm: physical height of the active area.
        viewing_distance_mm: eye to screen centre, along the screen normal.
    """

    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    viewing_distance_mm: float

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px", "width_mm", "height_mm", "viewing_distance_mm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def mm_per_px_x(self) -> float:
        return self.width_mm / self.width_px

    @property
    def mm_per_px_y(self) -> float:
        return self.height_mm / self.height_px

    def px_to_mm(self, points_px: np.ndarray) -> np.ndarray:
        """Convert screen pixel coordinates to millimetres from the top-left."""
        points_px = np.asarray(points_px, dtype=float)
        scale = np.array([self.mm_per_px_x, self.mm_per_px_y])
        return points_px * scale

    def angular_distance_deg(self, a_px: np.ndarray, b_px: np.ndarray) -> np.ndarray:
        """Visual angle between two on-screen points, in degrees.

        Both points are projected to the screen plane in millimetres and the
        angle is taken at the eye, which is assumed to sit on the normal
        through the screen centre at ``viewing_distance_mm``.

        Using the true subtended angle rather than the small-angle
        approximation ``d / D`` matters at the edges of a large screen: at 30
        degrees eccentricity the approximation is already off by about 10%.
        """
        a_px = np.atleast_2d(np.asarray(a_px, dtype=float))
        b_px = np.atleast_2d(np.asarray(b_px, dtype=float))

        centre_px = np.array([self.width_px / 2.0, self.height_px / 2.0])
        eye = np.array([0.0, 0.0, -self.viewing_distance_mm])

        def to_eye_frame(points_px: np.ndarray) -> np.ndarray:
            offset_mm = self.px_to_mm(points_px - centre_px)
            zeros = np.zeros((offset_mm.shape[0], 1))
            return np.hstack([offset_mm, zeros]) - eye

        va = to_eye_frame(a_px)
        vb = to_eye_frame(b_px)

        na = va / np.linalg.norm(va, axis=1, keepdims=True)
        nb = vb / np.linalg.norm(vb, axis=1, keepdims=True)

        cos = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def deg_to_px_at_centre(self, degrees: float) -> float:
        """How many pixels one visual degree spans at the screen centre.

        Only valid near the centre — the mapping is nonlinear towards the
        edges. Used for sizing targets, not for scoring accuracy.
        """
        mm = self.viewing_distance_mm * math.tan(math.radians(degrees))
        return mm / self.mm_per_px_x
