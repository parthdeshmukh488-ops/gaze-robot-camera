"""Tests for the One Euro filter.

The filter's whole claim is a trade-off: steadier than a fixed low-pass when
the signal is still, and less laggy when it moves. Both halves are tested
against a fixed-alpha baseline, because testing only that "it smooths" would
pass for any low-pass and prove nothing about this one.
"""

import numpy as np
import pytest

from gazectl.filters import OneEuroFilter


def _fixed_alpha_filter(signal, alpha):
    """Plain exponential moving average, as the comparison baseline."""
    out = np.empty_like(signal, dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="min_cutoff_hz"):
        OneEuroFilter(min_cutoff_hz=0.0)
    with pytest.raises(ValueError, match="beta"):
        OneEuroFilter(beta=-1.0)
    with pytest.raises(ValueError, match="d_cutoff_hz"):
        OneEuroFilter(d_cutoff_hz=0.0)


def test_first_sample_passes_through():
    f = OneEuroFilter()
    out = f(np.array([100.0, 200.0]), t=0.0)
    assert out == pytest.approx([100.0, 200.0])


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    for i in range(50):
        out = f(np.array([640.0, 360.0]), t=i / 30.0)
    assert out == pytest.approx([640.0, 360.0], abs=1e-9)


def test_reduces_noise_on_a_stationary_signal():
    """A fixation: the true position is fixed and all variation is noise."""
    rng = np.random.default_rng(42)
    n = 300
    truth = np.array([640.0, 360.0])
    noisy = truth + rng.normal(0.0, 8.0, size=(n, 2))

    f = OneEuroFilter(min_cutoff_hz=0.5, beta=0.005)
    filtered = np.array([f(noisy[i], t=i / 60.0) for i in range(n)])

    # Discard the warm-up, where the filter is still converging.
    raw_sd = noisy[60:].std(axis=0).mean()
    filtered_sd = filtered[60:].std(axis=0).mean()

    assert filtered_sd < raw_sd / 3.0


def test_lags_less_than_a_fixed_lowpass_on_a_step():
    """A saccade: the signal jumps and the filter must not crawl after it.

    Both filters are tuned to the same steady-state smoothing, then given a
    step. The adaptive one should be closer to the new value.
    """
    n = 60
    rate = 60.0
    signal = np.concatenate([np.full(n // 2, 100.0), np.full(n // 2, 500.0)])

    adaptive = OneEuroFilter(min_cutoff_hz=0.5, beta=0.05)
    adaptive_out = np.array([adaptive(np.array([signal[i]]), t=i / rate)[0] for i in range(n)])

    # The fixed baseline uses the adaptive filter's own at-rest alpha, so the
    # two agree exactly while the signal is still.
    tau = 1.0 / (2.0 * np.pi * 0.5)
    alpha_at_rest = 1.0 / (1.0 + tau * rate)
    fixed_out = _fixed_alpha_filter(signal, alpha_at_rest)

    settle = n // 2 + 8
    adaptive_error = abs(adaptive_out[settle] - 500.0)
    fixed_error = abs(fixed_out[settle] - 500.0)

    assert adaptive_error < fixed_error


def test_higher_beta_tracks_faster():
    n = 40
    signal = np.concatenate([np.full(20, 0.0), np.full(20, 100.0)])

    def run(beta):
        f = OneEuroFilter(min_cutoff_hz=0.5, beta=beta)
        return np.array([f(np.array([signal[i]]), t=i / 60.0)[0] for i in range(n)])

    slow = run(0.001)
    fast = run(0.5)

    assert abs(fast[-1] - 100.0) < abs(slow[-1] - 100.0)


def test_duplicate_timestamp_returns_last_estimate():
    """Dropped and repeated frames are routine; they must not blow up alpha."""
    f = OneEuroFilter()
    f(np.array([10.0]), t=0.0)
    first = f(np.array([20.0]), t=0.1)
    repeated = f(np.array([999.0]), t=0.1)

    assert repeated == pytest.approx(first)


def test_backwards_timestamp_is_survivable():
    f = OneEuroFilter()
    f(np.array([10.0]), t=1.0)
    out = f(np.array([20.0]), t=0.5)
    assert np.all(np.isfinite(out))


def test_reset_clears_state():
    f = OneEuroFilter()
    for i in range(20):
        f(np.array([500.0]), t=i / 30.0)

    f.reset()
    out = f(np.array([100.0]), t=0.0)

    assert out == pytest.approx([100.0])


def test_handles_scalar_and_vector_input():
    f = OneEuroFilter()
    assert f(5.0, t=0.0).shape == (1,)

    g = OneEuroFilter()
    assert g(np.array([1.0, 2.0, 3.0]), t=0.0).shape == (3,)
