"""The jitter-versus-lag trade-off, measured.

A gaze cursor has two ways of feeling wrong. It shivers while you hold still,
or it trails behind when you look somewhere else. Smoothing trades one for the
other, and a fixed low-pass filter has to pick a single point on that trade-off
and live with it everywhere.

This example builds a trace with both regimes in it — long fixations joined by
fast saccades — and measures both costs for a fixed filter and for the One
Euro filter, so the trade-off is a table rather than an assertion.
"""

import numpy as np

from gazectl.filters import OneEuroFilter
from gazectl.geometry import ScreenGeometry

SCREEN = ScreenGeometry(1920, 1080, 531.0, 299.0, 600.0)
RATE_HZ = 60.0
NOISE_PX = 6.0
SEED = 20260918

FIXATION_POINTS = [
    (400.0, 300.0),
    (1500.0, 300.0),
    (1500.0, 800.0),
    (400.0, 800.0),
    (960.0, 540.0),
]
FIXATION_SAMPLES = 90  # 1.5 seconds each
JITTER_SKIP = 45  # discard the settling transient before measuring jitter


def build_trace(rng):
    """Truth and a noisy observation of it.

    The saccades are instantaneous by construction. A real saccade takes 30-80
    ms, but making it a step is the harder test: any lag a filter introduces
    shows up undiluted.
    """
    truth = np.vstack(
        [np.tile(np.array(p, dtype=float), (FIXATION_SAMPLES, 1)) for p in FIXATION_POINTS]
    )
    observed = truth + rng.normal(0.0, NOISE_PX, size=truth.shape)
    timestamps = np.arange(len(truth)) / RATE_HZ
    return truth, observed, timestamps


def fixed_lowpass(observed, alpha):
    out = np.empty_like(observed)
    out[0] = observed[0]
    for i in range(1, len(observed)):
        out[i] = alpha * observed[i] + (1 - alpha) * out[i - 1]
    return out


def apply_one_euro(observed, timestamps, *, min_cutoff, beta):
    f = OneEuroFilter(min_cutoff_hz=min_cutoff, beta=beta)
    return np.array([f(observed[i], t=timestamps[i]) for i in range(len(observed))])


def jitter_deg(filtered, truth):
    """Variability during the steady part of each fixation, in degrees.

    Measured as dispersion about the segment's *own* mean, not about the true
    target. That distinction matters: a heavily smoothed signal that has not
    finished settling still sits away from the target, and scoring it against
    the target would charge that leftover lag to jitter and report the heavy
    filter as the noisier one — the opposite of what it is.
    """
    per_fixation = []
    for k in range(len(FIXATION_POINTS)):
        lo = k * FIXATION_SAMPLES + JITTER_SKIP
        hi = (k + 1) * FIXATION_SAMPLES
        segment = filtered[lo:hi]
        centre = np.tile(segment.mean(axis=0), (len(segment), 1))
        per_fixation.append(np.mean(SCREEN.angular_distance_deg(segment, centre) ** 2))
    return float(np.sqrt(np.mean(per_fixation)))


def settling_samples(filtered, truth, tolerance_deg=0.5):
    """How many samples after each saccade until the output is within tolerance."""
    counts = []
    for k in range(1, len(FIXATION_POINTS)):
        start = k * FIXATION_SAMPLES
        end = (k + 1) * FIXATION_SAMPLES
        target = np.tile(truth[start], (end - start, 1))
        error = SCREEN.angular_distance_deg(filtered[start:end], target)
        within = np.flatnonzero(error <= tolerance_deg)
        counts.append(int(within[0]) if within.size else end - start)
    return float(np.mean(counts))


def main() -> None:
    rng = np.random.default_rng(SEED)
    truth, observed, timestamps = build_trace(rng)

    print(
        f"Gaze trace: {len(FIXATION_POINTS)} fixations of "
        f"{FIXATION_SAMPLES / RATE_HZ:.1f} s joined by instantaneous saccades"
    )
    print(f"  sample rate            {RATE_HZ:.0f} Hz")
    print(f"  observation noise      {NOISE_PX:.0f} px SD")
    print(f"  jitter measured after  {JITTER_SKIP} samples of each fixation")
    print()

    rows = []
    rows.append(("unfiltered", observed))

    for alpha, label in ((0.60, "fixed low-pass, light"), (0.12, "fixed low-pass, heavy")):
        rows.append((f"{label} (a={alpha})", fixed_lowpass(observed, alpha)))

    rows.append(
        (
            "One Euro (fc=0.7, b=0.02)",
            apply_one_euro(observed, timestamps, min_cutoff=0.7, beta=0.02),
        )
    )

    print(f"{'filter':<32} {'jitter (deg)':>14} {'settling (samples)':>20}")
    print("-" * 68)
    for label, signal in rows:
        print(
            f"{label:<32} {jitter_deg(signal, truth):>14.3f} {settling_samples(signal, truth):>20.1f}"
        )

    print()
    print("The heavy fixed filter buys its low jitter with settling time it pays")
    print("on every saccade. The One Euro filter is near the heavy filter while")
    print("the eye is still and near the light one while it moves, which is the")
    print("whole reason to prefer it for an interactive cursor.")


if __name__ == "__main__":
    main()
