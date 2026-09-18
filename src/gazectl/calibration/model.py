"""Mapping gaze features to screen coordinates, fitted per user.

A webcam gives you where the iris sits inside the eye opening. It does not give
you where the person is looking — that depends on the eye's own geometry, where
the head is, and where the screen is. The usual answer is not to model any of
that but to fit a low-order polynomial per user from a short calibration
routine, which absorbs all of it into the coefficients.

Two things this module insists on:

- **A small ridge penalty by default.** A full second-order fit has 6
  coefficients per axis against the 9 targets of the standard routine, so the
  normal equations are close to ill-conditioned and near-duplicate calibration
  samples can make them worse. The default of 1e-3 is there for conditioning,
  not for accuracy: on the synthetic generator in the test suite the
  cross-validated error is flat from zero up to about 1e-2, and a large penalty
  makes things worse, not better. Do not reach for it as an accuracy knob.
- **Feature standardisation.** The raw features have wildly different scales
  (a normalised iris offset near 0.1, a head yaw in radians). Without
  standardising, a single ridge penalty means something different for each one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def polynomial_features(features: np.ndarray, degree: int = 2) -> np.ndarray:
    """Expand raw features into a polynomial design matrix with a bias column.

    Degree 1 gives [1, x_1 ... x_d]; degree 2 adds every square and pairwise
    product. Degree 2 is the right default: it captures the curvature of the
    eye-to-screen mapping, and degree 3 overfits a nine-point routine badly.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    n, d = features.shape

    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2, got {degree}")

    columns = [np.ones((n, 1))]
    columns.append(features)

    if degree == 2:
        for i in range(d):
            for j in range(i, d):
                columns.append((features[:, i] * features[:, j]).reshape(-1, 1))

    return np.hstack(columns)


@dataclass
class CalibrationModel:
    """A fitted per-user gaze-to-screen mapping.

    Args:
        degree: polynomial degree of the design matrix.
        ridge: L2 penalty. The bias column is never penalised.
    """

    degree: int = 2
    ridge: float = 1e-3

    _coefficients: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._coefficients is not None

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(features, dtype=float)) - self._mean) / self._scale

    def fit(self, features: np.ndarray, targets_px: np.ndarray) -> CalibrationModel:
        """Fit the mapping.

        Args:
            features: (n, d) gaze features, one row per calibration sample.
            targets_px: (n, 2) the on-screen point the user was looking at.
        """
        features = np.atleast_2d(np.asarray(features, dtype=float))
        targets_px = np.atleast_2d(np.asarray(targets_px, dtype=float))

        if features.shape[0] != targets_px.shape[0]:
            raise ValueError("features and targets_px must have the same length")
        if targets_px.shape[1] != 2:
            raise ValueError(f"targets_px must be (n, 2), got {targets_px.shape}")

        n_params = polynomial_features(np.zeros((1, features.shape[1])), self.degree).shape[1]
        if features.shape[0] < n_params:
            raise ValueError(
                f"need at least {n_params} samples for a degree-{self.degree} fit on "
                f"{features.shape[1]} features, got {features.shape[0]}"
            )

        self._mean = features.mean(axis=0)
        scale = features.std(axis=0)
        # A feature that never varies carries no information; a scale of 1.0
        # leaves its standardised column at zero rather than producing inf.
        self._scale = np.where(scale < 1e-12, 1.0, scale)

        design = polynomial_features(self._standardise(features), self.degree)

        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0  # never penalise the bias

        gram = design.T @ design + penalty
        self._coefficients = np.linalg.solve(gram, design.T @ targets_px)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Map gaze features to screen pixels. Returns (n, 2)."""
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        design = polynomial_features(self._standardise(features), self.degree)
        return design @ self._coefficients


def leave_one_target_out(
    features: np.ndarray,
    targets_px: np.ndarray,
    target_ids: np.ndarray,
    *,
    degree: int = 2,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Cross-validated predictions, holding out one calibration target at a time.

    Holding out individual *samples* would be meaningless here: samples from the
    same target are nearly identical, so a sample-wise split leaks the answer
    and reports an accuracy the model does not have. Holding out a whole target
    asks the question that matters — how well does this fit generalise to a
    screen position it never saw?

    Returns:
        (n, 2) predicted screen points, each from a model that never saw its
        own target.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    targets_px = np.atleast_2d(np.asarray(targets_px, dtype=float))
    target_ids = np.asarray(target_ids)

    predictions = np.full_like(targets_px, np.nan, dtype=float)

    for held_out in np.unique(target_ids):
        test = target_ids == held_out
        train = ~test
        if train.sum() == 0:
            continue
        model = CalibrationModel(degree=degree, ridge=ridge)
        model.fit(features[train], targets_px[train])
        predictions[test] = model.predict(features[test])

    return predictions
