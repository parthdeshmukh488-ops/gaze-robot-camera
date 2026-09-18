"""MediaPipe Face Landmarker adapter: frames in, gaze features out.

This is the only module that knows what a camera is. Everything downstream
takes a feature vector and a validity flag, which is what lets the rest of the
package be tested without hardware.

**What the features are.** Not a gaze direction — a description of the eye's
configuration, which the per-user calibration fit then maps to the screen:

- the iris centre's offset inside its own eye opening, per eye, normalised by
  the eye's width and height
- head yaw, pitch and roll

Normalising the iris offset by the eye opening is what makes the features
survive the user moving towards or away from the camera: a raw pixel offset
doubles when the face doubles in size on the sensor, and the calibration would
be invalidated by leaning forward.

**Why head pose is in the feature vector rather than corrected for
separately.** The same iris offset means a different gaze point depending on
where the head is pointing. Feeding the head angles in as features lets the
polynomial fit learn that interaction from the calibration data, instead of
requiring a hand-derived correction that would need the eye's rotation centre —
a quantity a webcam cannot measure.

Requires the ``tracking`` extra: ``pip install "gazectl[tracking]"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe's refined face mesh returns 478 landmarks. The last ten are the
# irises, which only exist when refine_landmarks is enabled.
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)

# Eye corners, used to normalise the iris offset against the eye opening.
LEFT_EYE_CORNERS = (33, 133)  # outer, inner
RIGHT_EYE_CORNERS = (362, 263)  # inner, outer
LEFT_EYE_LIDS = (159, 145)  # upper, lower
RIGHT_EYE_LIDS = (386, 374)

# A coarse canonical face in millimetres, for solvePnP. Only the head angles
# are used downstream, so an approximate model is adequate; a per-subject one
# would be needed if absolute head translation mattered.
CANONICAL_FACE_MM = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (-43.3, 32.7, -26.0),  # left eye outer corner
        (43.3, 32.7, -26.0),  # right eye outer corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)
CANONICAL_INDICES = (1, 152, 33, 263, 61, 291)

FEATURE_NAMES = (
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "head_yaw",
    "head_pitch",
    "head_roll",
)


@dataclass(frozen=True)
class GazeFeatures:
    """One frame's worth of features, plus whether they are usable."""

    features: np.ndarray  # shape (7,), see FEATURE_NAMES
    valid: bool
    timestamp: float
    eye_openness: float = 1.0

    def as_dict(self) -> dict:
        return dict(zip(FEATURE_NAMES, self.features.tolist(), strict=True))


def _normalised_iris_offset(
    landmarks: np.ndarray, iris_ids, corner_ids, lid_ids
) -> tuple[np.ndarray, float]:
    """Iris centre inside its eye opening, in [-0.5, 0.5]-ish units.

    Returns the offset and the eye's aspect-based openness, which is used to
    reject blinks. A closed eye still yields an iris position from the mesh —
    MediaPipe interpolates it — and that position is meaningless, so openness
    is the signal that decides validity rather than anything about the iris.
    """
    iris = landmarks[list(iris_ids)].mean(axis=0)

    outer, inner = landmarks[corner_ids[0]], landmarks[corner_ids[1]]
    upper, lower = landmarks[lid_ids[0]], landmarks[lid_ids[1]]

    centre = (outer + inner) / 2.0
    width = float(np.linalg.norm(inner - outer))
    height = float(np.linalg.norm(upper - lower))

    if width < 1e-6:
        return np.zeros(2), 0.0

    offset = np.array([(iris[0] - centre[0]) / width, (iris[1] - centre[1]) / width])
    return offset, height / width


def _head_angles(landmarks_px: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Yaw, pitch and roll in radians, via solvePnP on six canonical points."""
    import cv2

    width, height = image_size
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    image_points = landmarks_px[list(CANONICAL_INDICES)].astype(np.float64)

    ok, rotation, _ = cv2.solvePnP(
        CANONICAL_FACE_MM,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return np.zeros(3)

    matrix, _ = cv2.Rodrigues(rotation)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))

    if sy > 1e-6:
        pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        # Gimbal-lock branch: roll and yaw are not separable, so roll is
        # pinned to zero rather than returning an arbitrary split.
        pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = 0.0

    return np.array([yaw, pitch, roll])


class MediaPipeGazeSource:
    """Extracts gaze features from frames using MediaPipe Face Landmarker.

    Args:
        min_eye_openness: below this height/width ratio the frame is marked
            invalid. The default rejects blinks without rejecting people whose
            eyes are simply narrow, but it is the first thing to tune per user
            if data loss looks high.
        min_detection_confidence: passed through to MediaPipe.
    """

    def __init__(
        self,
        *,
        min_eye_openness: float = 0.15,
        min_detection_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'MediaPipeGazeSource needs the tracking extra: pip install "gazectl[tracking]"'
            ) from exc

        self.min_eye_openness = min_eye_openness
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # required for the iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )

    def process(self, frame_rgb: np.ndarray, timestamp: float) -> GazeFeatures:
        """Extract features from one RGB frame.

        A frame with no face, or with the eyes closed, comes back with
        ``valid=False`` and zeroed features. It is never silently dropped:
        downstream, a missing sample is what stops the robot, so it has to
        arrive rather than simply not arrive.
        """
        height, width = frame_rgb.shape[:2]
        result = self._mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return GazeFeatures(np.zeros(len(FEATURE_NAMES)), False, timestamp, 0.0)

        face = result.multi_face_landmarks[0]
        normalised = np.array([(lm.x, lm.y) for lm in face.landmark])
        pixels = normalised * np.array([width, height])

        left_offset, left_open = _normalised_iris_offset(
            normalised, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS
        )
        right_offset, right_open = _normalised_iris_offset(
            normalised, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS
        )
        openness = min(left_open, right_open)

        if openness < self.min_eye_openness:
            return GazeFeatures(np.zeros(len(FEATURE_NAMES)), False, timestamp, openness)

        features = np.concatenate(
            [left_offset, right_offset, _head_angles(pixels, (width, height))]
        )
        return GazeFeatures(features, True, timestamp, openness)

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> MediaPipeGazeSource:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
