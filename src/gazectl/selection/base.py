"""Shared vocabulary for the selection techniques compared in the user study.

Every technique is a state machine fed one gaze sample at a time and returning
whatever happened on that sample. Keeping them behind one interface is what
makes the study fair: the same trace can be replayed through all three, so a
difference in the results is a difference between techniques rather than
between three separately written event loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class SelectionEvent(Enum):
    """What a technique reports for a single sample."""

    NONE = "none"
    ENTERED = "entered"  # gaze arrived on a target
    LEFT = "left"  # gaze left the target it was on
    SELECTED = "selected"  # a selection was committed
    CANCELLED = "cancelled"  # a pending selection was abandoned


@dataclass(frozen=True)
class Target:
    """A rectangular, axis-aligned selectable region in screen pixels."""

    id: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, point_px: np.ndarray) -> bool:
        px, py = float(point_px[0]), float(point_px[1])
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)

    @property
    def centre_px(self) -> np.ndarray:
        return np.array([self.x + self.width / 2.0, self.y + self.height / 2.0])


@dataclass
class SelectionResult:
    """Outcome of feeding one sample to a technique."""

    event: SelectionEvent = SelectionEvent.NONE
    target_id: str | None = None
    progress: float = 0.0  # 0..1, for drawing dwell feedback
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class SelectionTechnique(Protocol):
    """The interface every technique implements."""

    def update(self, point_px: np.ndarray, t: float, *, valid: bool = True) -> SelectionResult:
        """Feed one gaze sample.

        Args:
            point_px: filtered gaze point in screen pixels.
            t: timestamp in seconds.
            valid: False when tracking was lost on this sample.
        """
        ...

    def reset(self) -> None:
        """Drop all state, as between trials."""
        ...


def target_at(targets: list[Target], point_px: np.ndarray) -> Target | None:
    """The first target containing the point, or None.

    Targets are tested in order, so overlapping targets resolve to whichever
    was registered first. The study layout has no overlaps; this is only
    defined behaviour for the case where a caller builds one.
    """
    for target in targets:
        if target.contains(point_px):
            return target
    return None
