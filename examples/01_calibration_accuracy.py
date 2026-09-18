"""Calibration accuracy, scored honestly.

Runs the nine-point routine on a synthetic eye whose true gaze is known, then
reports accuracy two ways: on the calibration data itself, and cross-validated
by holding out one target at a time.

The gap between those two numbers is the point of the example. Reporting the
first as "our accuracy" is the most common way an eye-tracking result is
overstated, and the second is the number that predicts what happens when the
user looks somewhere the calibration did not cover.

The eye here is synthetic, so these are not measurements of a real tracker.
They are a demonstration that the evaluation machinery reports what it should.
"""

import numpy as np

from gazectl.calibration import (
    CalibrationModel,
    accuracy,
    data_loss_rate,
    leave_one_target_out,
    precision_rms_s2s,
    precision_sd,
)
from gazectl.geometry import ScreenGeometry

SCREEN = ScreenGeometry(
    width_px=1920,
    height_px=1080,
    width_mm=531.0,
    height_mm=299.0,
    viewing_distance_mm=600.0,
)

SAMPLES_PER_TARGET = 30
FEATURE_NOISE = 0.012
DROPOUT_RATE = 0.04
SEED = 20260918


def nine_point_grid(margin: float = 0.1) -> np.ndarray:
    xs = np.array([margin, 0.5, 1.0 - margin]) * SCREEN.width_px
    ys = np.array([margin, 0.5, 1.0 - margin]) * SCREEN.height_px
    return np.array([[x, y] for y in ys for x in xs])


def synthetic_eye(targets_px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A mildly nonlinear, noisy map from screen position to iris offset.

    The cubic terms are what make this worth fitting with a quadratic rather
    than a plane: a linear model leaves visible structure in the residual.
    """
    u = targets_px[:, 0] / SCREEN.width_px - 0.5
    v = targets_px[:, 1] / SCREEN.height_px - 0.5

    fx = 0.30 * u + 0.08 * u**3 + 0.02 * u * v
    fy = 0.28 * v + 0.06 * v**3 - 0.015 * u * v

    features = np.column_stack([fx, fy])
    return features + rng.normal(0.0, FEATURE_NOISE, size=features.shape)


def main() -> None:
    rng = np.random.default_rng(SEED)

    grid = nine_point_grid()
    targets = np.repeat(grid, SAMPLES_PER_TARGET, axis=0)
    target_ids = np.repeat(np.arange(len(grid)), SAMPLES_PER_TARGET)
    features = synthetic_eye(targets, rng)

    valid = rng.random(len(targets)) > DROPOUT_RATE

    print("Nine-point calibration on a synthetic eye")
    print(f"  targets                {len(grid)}")
    print(f"  samples per target     {SAMPLES_PER_TARGET}")
    print(f"  data loss              {data_loss_rate(valid) * 100:.1f}%")
    print()

    model = CalibrationModel(degree=2, ridge=1e-3)
    model.fit(features[valid], targets[valid])

    in_sample = accuracy(model.predict(features[valid]), targets[valid], SCREEN)
    cross_validated = accuracy(
        leave_one_target_out(features[valid], targets[valid], target_ids[valid]),
        targets[valid],
        SCREEN,
    )

    print("Accuracy (degrees of visual angle)")
    print(
        f"  on calibration data    {in_sample.mean_deg:.3f} mean, {in_sample.p95_deg:.3f} p95"
    )
    print(
        f"  held-out target        {cross_validated.mean_deg:.3f} mean, "
        f"{cross_validated.p95_deg:.3f} p95"
    )
    print(
        f"  overstatement          {cross_validated.mean_deg / in_sample.mean_deg:.2f}x "
        "if the first number is quoted"
    )
    print()

    # Precision is measured on a held-still gaze, which is a different
    # experiment from accuracy: the eye does not move, so any variation in the
    # reported point is the tracker's own noise.
    centre = np.array([SCREEN.width_px / 2.0, SCREEN.height_px / 2.0])
    held_still_features = synthetic_eye(np.tile(centre, (600, 1)), rng)
    held_still = model.predict(held_still_features)

    print("Precision at the screen centre (degrees)")
    print(f"  sample-to-sample RMS   {precision_rms_s2s(held_still, SCREEN):.3f}")
    print(f"  dispersion SD          {precision_sd(held_still, SCREEN):.3f}")
    print()
    print("  The two differ by about sqrt(2) for uncorrelated noise. Quoting the")
    print("  smaller one against someone else's larger one makes a tracker look")
    print("  roughly 40% better than it is.")


if __name__ == "__main__":
    main()
