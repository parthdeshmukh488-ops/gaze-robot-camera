"""Tests for the calibration fit and its accuracy/precision metrics.

The interesting tests here are the ones about generalisation. A calibration
model that is only checked on its own calibration data will always look
excellent, which is exactly the failure this module is designed to prevent.
"""

import numpy as np
import pytest

from gazectl.calibration import (
    CalibrationModel,
    accuracy,
    data_loss_rate,
    leave_one_target_out,
    polynomial_features,
    precision_rms_s2s,
    precision_sd,
)
from gazectl.geometry import ScreenGeometry


@pytest.fixture
def screen():
    return ScreenGeometry(1920, 1080, 531.0, 299.0, 600.0)


def _nine_point_grid(width=1920, height=1080, margin=0.1):
    xs = np.array([margin, 0.5, 1 - margin]) * width
    ys = np.array([margin, 0.5, 1 - margin]) * height
    return np.array([[x, y] for y in ys for x in xs])


def _synthetic_features(targets_px, *, noise=0.0, seed=0):
    """A plausible nonlinear eye-to-screen relationship, inverted.

    The mapping from screen position to a normalised iris offset is
    monotonic and mildly curved. Generating features this way means the
    polynomial fit has something real to recover rather than a straight line.
    """
    rng = np.random.default_rng(seed)
    u = targets_px[:, 0] / 1920.0 - 0.5
    v = targets_px[:, 1] / 1080.0 - 0.5

    fx = 0.30 * u + 0.08 * u**3 + 0.02 * u * v
    fy = 0.28 * v + 0.06 * v**3 - 0.015 * u * v

    features = np.column_stack([fx, fy])
    if noise:
        features = features + rng.normal(0.0, noise, size=features.shape)
    return features


def test_polynomial_features_shape_and_bias():
    x = np.array([[2.0, 3.0]])

    linear = polynomial_features(x, degree=1)
    assert linear.shape == (1, 3)  # bias + 2
    assert linear[0, 0] == 1.0

    quad = polynomial_features(x, degree=2)
    assert quad.shape == (1, 6)  # bias + 2 + {xx, xy, yy}
    assert quad[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])


def test_polynomial_features_rejects_unsupported_degree():
    with pytest.raises(ValueError, match="degree"):
        polynomial_features(np.zeros((1, 2)), degree=3)


def test_fit_recovers_a_clean_mapping(screen):
    targets = np.repeat(_nine_point_grid(), 20, axis=0)
    features = _synthetic_features(targets)

    model = CalibrationModel(degree=2, ridge=1e-6).fit(features, targets)
    report = accuracy(model.predict(features), targets, screen)

    assert report.mean_deg < 0.5


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        CalibrationModel().fit(np.zeros((10, 2)), np.zeros((8, 2)))


def test_fit_requires_enough_samples():
    """Six coefficients per axis cannot be fitted from four points."""
    with pytest.raises(ValueError, match="at least"):
        CalibrationModel(degree=2).fit(np.zeros((4, 2)), np.zeros((4, 2)))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        CalibrationModel().predict(np.zeros((1, 2)))


def test_constant_feature_does_not_produce_nan():
    """A feature that never varies has zero scale; it must not divide by zero."""
    targets = np.repeat(_nine_point_grid(), 5, axis=0)
    features = _synthetic_features(targets)
    features = np.column_stack([features, np.full(len(features), 3.0)])

    model = CalibrationModel(degree=2).fit(features, targets)
    predictions = model.predict(features)

    assert np.all(np.isfinite(predictions))


def test_held_out_target_is_worse_than_training_fit(screen):
    """The point of leave-one-target-out.

    Scoring on the calibration data flatters the model. The cross-validated
    error is the honest one and must be the larger of the two.
    """
    grid = _nine_point_grid()
    targets = np.repeat(grid, 15, axis=0)
    target_ids = np.repeat(np.arange(len(grid)), 15)
    features = _synthetic_features(targets, noise=0.004, seed=3)

    model = CalibrationModel(degree=2, ridge=1e-4).fit(features, targets)
    in_sample = accuracy(model.predict(features), targets, screen).mean_deg

    cv_predictions = leave_one_target_out(features, targets, target_ids, ridge=1e-4)
    cross_validated = accuracy(cv_predictions, targets, screen).mean_deg

    assert cross_validated > in_sample


def test_default_ridge_costs_almost_nothing(screen):
    """The default penalty must not be paying for itself with accuracy.

    Measured rather than assumed: on this generator the cross-validated error
    is flat from an effectively-zero penalty up to about 1e-2, so the default
    of 1e-3 buys conditioning without measurably degrading the fit. If that
    stops being true, this test is the thing that notices.
    """
    grid = _nine_point_grid()
    targets = np.repeat(grid, 10, axis=0)
    target_ids = np.repeat(np.arange(len(grid)), 10)
    features = _synthetic_features(targets, noise=0.02, seed=11)

    unregularised = accuracy(
        leave_one_target_out(features, targets, target_ids, ridge=1e-12), targets, screen
    ).mean_deg
    default = accuracy(
        leave_one_target_out(features, targets, target_ids, ridge=1e-3), targets, screen
    ).mean_deg

    assert default == pytest.approx(unregularised, rel=0.01)


def test_large_ridge_shrinks_towards_the_target_centroid(screen):
    """What the penalty actually does, stated as a test.

    Ridge shrinks the non-bias coefficients towards zero; since the bias is
    never penalised, the fit collapses towards the mean of the calibration
    targets. Confirming the direction of the effect is worth more than
    asserting it improves accuracy, which on this generator it does not.
    """
    grid = _nine_point_grid()
    targets = np.repeat(grid, 10, axis=0)
    features = _synthetic_features(targets, noise=0.01, seed=4)
    centroid = targets.mean(axis=0)

    light = CalibrationModel(degree=2, ridge=1e-6).fit(features, targets).predict(features)
    heavy = CalibrationModel(degree=2, ridge=1e4).fit(features, targets).predict(features)

    spread_light = np.linalg.norm(light - centroid, axis=1).mean()
    spread_heavy = np.linalg.norm(heavy - centroid, axis=1).mean()

    assert spread_heavy < spread_light / 10.0


def test_accuracy_is_zero_for_perfect_predictions(screen):
    targets = _nine_point_grid()
    report = accuracy(targets, targets, screen)
    assert report.mean_deg == pytest.approx(0.0, abs=1e-9)
    assert report.n_samples == len(targets)


def test_accuracy_reports_a_tail(screen):
    """Why the median and p95 are reported alongside the mean.

    Ten bad predictions in a hundred pull the mean well above the median. A
    tracker described by its mean alone looks worse than it typically is; one
    described by its median alone hides that it sometimes fails badly.
    """
    targets = np.tile(np.array([960.0, 540.0]), (100, 1))
    predictions = targets.copy()
    predictions[:90] += 5.0
    predictions[90:] += 300.0

    report = accuracy(predictions, targets, screen)

    assert report.median_deg < report.mean_deg < report.p95_deg


def test_accuracy_ignores_non_finite_pairs(screen):
    targets = np.tile(np.array([960.0, 540.0]), (10, 1))
    predictions = targets.copy()
    predictions[3] = np.nan

    report = accuracy(predictions, targets, screen)
    assert report.n_samples == 9


def test_accuracy_raises_when_nothing_is_finite(screen):
    targets = np.full((4, 2), np.nan)
    with pytest.raises(ValueError, match="no finite"):
        accuracy(targets, targets, screen)


def test_precision_is_zero_for_a_perfectly_still_signal(screen):
    points = np.tile(np.array([960.0, 540.0]), (50, 1))
    assert precision_rms_s2s(points, screen) == pytest.approx(0.0, abs=1e-12)
    assert precision_sd(points, screen) == pytest.approx(0.0, abs=1e-12)


def test_precision_grows_with_noise(screen):
    rng = np.random.default_rng(5)
    centre = np.array([960.0, 540.0])

    quiet = centre + rng.normal(0.0, 1.0, size=(200, 2))
    loud = centre + rng.normal(0.0, 10.0, size=(200, 2))

    assert precision_rms_s2s(loud, screen) > precision_rms_s2s(quiet, screen)
    assert precision_sd(loud, screen) > precision_sd(quiet, screen)


def test_sample_to_sample_exceeds_dispersion_for_white_noise(screen):
    """Why the two definitions are not interchangeable.

    For uncorrelated noise the sample-to-sample figure is about sqrt(2) times
    the dispersion figure, so quoting one against the other misstates a
    tracker by roughly 40%.
    """
    rng = np.random.default_rng(9)
    points = np.array([960.0, 540.0]) + rng.normal(0.0, 5.0, size=(4000, 2))

    ratio = precision_rms_s2s(points, screen) / precision_sd(points, screen)
    assert ratio == pytest.approx(np.sqrt(2.0), rel=0.10)


def test_precision_needs_two_samples(screen):
    with pytest.raises(ValueError, match="at least two"):
        precision_rms_s2s(np.array([[1.0, 2.0]]), screen)


def test_data_loss_rate():
    assert data_loss_rate(np.ones(10, dtype=bool)) == 0.0
    assert data_loss_rate(np.zeros(10, dtype=bool)) == 1.0
    assert data_loss_rate(np.array([True, True, False, False])) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="non-empty"):
        data_loss_rate(np.array([], dtype=bool))
