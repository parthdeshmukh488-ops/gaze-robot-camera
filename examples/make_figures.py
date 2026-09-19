"""Regenerate the figures in docs/figures/.

Kept separate from the examples that print the tables, for the same reason the
simulation and plotting are separate everywhere else here: figures get restyled
far more often than the numbers change, and a cosmetic edit should not be able
to quietly redraw a figure from a different set of random numbers than the
README quotes. Both use the same seed and the same generators.

    python examples/make_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gazectl.filters import OneEuroFilter, detect_fixations
from gazectl.geometry import ScreenGeometry
from gazectl.robot import SafetyFilter, WorkspaceLimits
from gazectl.selection import DwellSelector, LookAndConfirmSelector, SelectionEvent, Target

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"
SCREEN = ScreenGeometry(1920, 1080, 531.0, 299.0, 600.0)
SEED = 20260918

INK = "#1b1f24"
MUTED = "#8c959f"
RAW = "#bcc4cc"
ACCENT = "#0969da"
WARM = "#bc4c00"
GOOD = "#1a7f37"
BAD = "#cf222e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    }
)


def fixed_lowpass(signal, alpha):
    out = np.empty_like(signal, dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


# --------------------------------------------------------------- figure 1


def figure_filters():
    """What jitter and lag actually look like on a gaze trace."""
    rng = np.random.default_rng(SEED)
    rate = 60.0
    points = [(400.0, 300.0), (1500.0, 300.0), (1500.0, 800.0), (600.0, 600.0)]
    per = 90

    truth = np.vstack([np.tile(np.array(p), (per, 1)) for p in points])
    observed = truth + rng.normal(0.0, 6.0, size=truth.shape)
    t = np.arange(len(truth)) / rate

    euro = OneEuroFilter(min_cutoff_hz=0.7, beta=0.02)
    one_euro = np.array([euro(observed[i], t=t[i]) for i in range(len(observed))])
    heavy = fixed_lowpass(observed, 0.12)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)

    for ax, (axis, label) in zip(axes, [(0, "horizontal"), (1, "vertical")], strict=True):
        ax.plot(t, observed[:, axis], color=RAW, lw=0.8, label="raw gaze", zorder=1)
        ax.plot(t, heavy[:, axis], color=WARM, lw=1.6, label="fixed low-pass (heavy)", zorder=2)
        ax.plot(t, one_euro[:, axis], color=ACCENT, lw=1.8, label="One Euro", zorder=3)
        ax.plot(
            t, truth[:, axis], color=INK, lw=1.0, ls=(0, (4, 3)), label="true gaze", zorder=4
        )
        ax.set_ylabel(f"{label} (px)")

    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    axes[1].set_xlabel("time (s)")

    for k in range(1, len(points)):
        for ax in axes:
            ax.axvline(k * per / rate, color=MUTED, lw=0.6, alpha=0.5)

    axes[1].annotate(
        "the heavy filter is still\ncatching up here",
        xy=(2 * per / rate + 0.22, 700),
        xytext=(2 * per / rate + 0.75, 430),
        fontsize=8.5,
        color=WARM,
        arrowprops={"arrowstyle": "->", "color": WARM, "lw": 1.0},
    )

    fig.suptitle(
        "Same smoothing at rest, no lag after a saccade",
        y=1.02,
        fontsize=12,
        fontweight="600",
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_filters.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2


def figure_selection():
    """Where each technique fires, against what the user actually wanted."""
    rng = np.random.default_rng(SEED)
    rate = 60.0

    targets = [
        Target(id="left", x=200.0, y=400.0, width=260.0, height=200.0),
        Target(id="middle", x=830.0, y=400.0, width=260.0, height=200.0),
        Target(id="right", x=1460.0, y=400.0, width=260.0, height=200.0),
    ]
    elsewhere = np.array([960.0, 100.0])

    script = [
        ("middle", 1.2, True),
        (None, 0.4, False),
        ("left", 1.0, False),
        (None, 0.3, False),
        ("right", 1.2, True),
        (None, 0.4, False),
        ("middle", 0.9, False),
        (None, 0.3, False),
        ("left", 1.2, True),
        (None, 0.5, False),
    ]

    points, times, intents, spans = [], [], [], []
    t = 0.0
    for target_id, duration, intended in script:
        centre = (
            next(x for x in targets if x.id == target_id).centre_px if target_id else elsewhere
        )
        start = t
        for _ in range(int(duration * rate)):
            points.append(centre + rng.normal(0.0, 8.0, size=2))
            times.append(t)
            intents.append((target_id, intended))
            t += 1.0 / rate
        spans.append((target_id, intended, start, t))

    points, times = np.array(points), np.array(times)
    wanted = [s for s in spans if s[1]]

    def run_dwell(dwell_s):
        sel = DwellSelector(targets, dwell_s=dwell_s, refractory_s=0.6)
        fired = []
        for i in range(len(points)):
            r = sel.update(points[i], times[i])
            if r.event is SelectionEvent.SELECTED:
                fired.append((r.target_id, times[i]))
        return fired

    def run_confirm():
        sel = LookAndConfirmSelector(targets, confirm_window_s=0.3, min_fixation_s=0.15)
        fired, pressed, previous, start = [], False, None, times[0]
        for i in range(len(points)):
            segment = intents[i]
            if segment != previous:
                previous, start, pressed = segment, times[i], False
            sel.update(points[i], times[i])
            if segment[1] and not pressed and times[i] - start >= 0.25:
                r = sel.confirm(times[i])
                pressed = True
                if r.event is SelectionEvent.SELECTED:
                    fired.append((r.target_id, times[i]))
        return fired

    rows = [
        ("dwell 0.5 s", run_dwell(0.5)),
        ("dwell 0.8 s", run_dwell(0.8)),
        ("dwell 1.2 s", run_dwell(1.2)),
        ("look-and-confirm", run_confirm()),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 3.6))

    for target_id, intended, start, end in spans:
        if target_id is None:
            continue
        ax.axvspan(
            start,
            end,
            color=GOOD if intended else MUTED,
            alpha=0.16 if intended else 0.09,
            lw=0,
        )
        ax.text(
            (start + end) / 2,
            len(rows) - 0.25,
            f"{target_id}\n{'wants it' if intended else 'just reading'}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=GOOD if intended else MUTED,
        )

    for row, (_label, fired) in enumerate(rows):
        y = len(rows) - 1 - row
        ax.axhline(y, color=MUTED, lw=0.5, alpha=0.4)

        # A selection is correct when it lands inside a span the scripted user
        # actually wanted, and names that span's target. Matching on target id
        # alone would credit a fire during a *reading* span against the later
        # span where the same target was genuinely wanted.
        satisfied = set()
        for target_id, when in fired:
            span = next((s for s in spans if s[2] <= when < s[3]), None)
            correct = (
                span is not None
                and span[1]
                and span[0] == target_id
                and span[2] not in satisfied
            )
            if correct:
                satisfied.add(span[2])

            ax.scatter(
                when,
                y,
                s=96,
                marker="o" if correct else "X",
                color=GOOD if correct else BAD,
                zorder=3,
                edgecolors="white",
                linewidths=1.0,
            )

        missed = sum(1 for w in wanted if w[2] not in satisfied)
        note = f"{missed} missed" if missed else ""
        ax.text(times[-1] + 0.12, y, note, va="center", fontsize=8, color=BAD)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, times[-1] + 1.0)
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=GOOD, label="intended selection"),
        plt.Line2D([], [], marker="X", ls="", color=BAD, label="unintended (Midas touch)"),
    ]
    ax.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    ax.set_title("Dwell has no setting that is right", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "02_selection.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3


def figure_safety():
    """What the safety layer does to a hostile gaze signal."""
    rate = 60.0
    n = 420
    t = np.arange(n) / rate

    limits = WorkspaceLimits(pan_min=-1.0, pan_max=1.0, tilt_min=-0.5, tilt_max=0.5)
    filt = SafetyFilter(limits, max_speed_rad_s=1.2, tracking_timeout_s=0.25)

    desired = np.zeros((n, 2))
    desired[:, 0] = 0.8 * np.sign(np.sin(2 * np.pi * 0.28 * t))  # square wave: hard steps
    desired[:, 1] = 0.35 * np.sin(2 * np.pi * 0.2 * t)
    desired[150:170, 0] = 6.0  # a wild outlier, far outside the workspace

    valid = np.ones(n, dtype=bool)
    valid[250:310] = False  # a one-second tracking dropout

    commanded = np.zeros((n, 2))
    holding = np.zeros(n, dtype=bool)
    for i in range(n):
        commanded[i] = filt.step(desired[i] if valid[i] else None, t=t[i], valid=valid[i])
        holding[i] = filt.holding

    fig, ax = plt.subplots(figsize=(9.6, 3.8))

    ax.axhspan(limits.pan_min, limits.pan_max, color=GOOD, alpha=0.07, lw=0)
    ax.axhline(limits.pan_max, color=GOOD, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(limits.pan_min, color=GOOD, lw=1.0, ls=(0, (4, 3)))
    ax.text(t[-1], limits.pan_max + 0.05, "workspace limit", ha="right", fontsize=8, color=GOOD)

    ax.fill_between(
        t, -1.6, 1.6, where=~valid, color=BAD, alpha=0.10, lw=0, label="tracking lost"
    )
    ax.fill_between(
        t, -1.6, 1.6, where=holding, color=WARM, alpha=0.16, lw=0, label="robot holding"
    )

    ax.plot(t, np.clip(desired[:, 0], -1.6, 1.6), color=RAW, lw=1.4, label="gaze asks for")
    ax.plot(t, commanded[:, 0], color=ACCENT, lw=2.0, label="commanded to robot")

    ax.annotate(
        "gaze jumps far outside\nthe workspace",
        xy=(150 / rate, 1.55),
        xytext=(150 / rate - 1.5, 1.15),
        fontsize=8.5,
        color=MUTED,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0},
    )
    ax.annotate(
        "rate limit turns\neach step into a ramp",
        xy=(0.95, 0.35),
        xytext=(0.35, -1.35),
        fontsize=8.5,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pan (rad)")
    ax.set_ylim(-1.6, 1.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Three gates: clamp, rate limit, watchdog", loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "03_safety.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4


def figure_fixations():
    """I-VT on a trace that includes a blink."""
    rng = np.random.default_rng(SEED)
    rate = 60.0
    spots = [(500.0, 350.0), (1350.0, 400.0), (900.0, 780.0), (1500.0, 800.0)]
    per = 75

    pts = np.vstack([np.tile(np.array(p), (per, 1)) for p in spots])
    pts = pts + rng.normal(0.0, 7.0, size=pts.shape)
    t = np.arange(len(pts)) / rate

    valid = np.ones(len(pts), dtype=bool)
    valid[110:128] = False  # a blink mid-fixation

    fixations = detect_fixations(pts, t, SCREEN, valid=valid)
    without_mask = detect_fixations(pts, t, SCREEN)

    # Against time, not against the screen: two fixations either side of a
    # blink sit at the same place, so a spatial plot draws them on top of each
    # other and the split -- the whole point -- becomes invisible.
    fig, ax = plt.subplots(figsize=(9.6, 3.8))

    ax.plot(t, pts[:, 0], color=RAW, lw=1.0, zorder=2, label="gaze x")

    blink_start, blink_end = t[110], t[127]
    ax.axvspan(blink_start, blink_end, color=BAD, alpha=0.14, lw=0, label="tracking lost")

    for fix in fixations:
        ax.axvspan(fix.start_time, fix.end_time, color=ACCENT, alpha=0.13, lw=0)
        ax.hlines(
            fix.centroid_px[0],
            fix.start_time,
            fix.end_time,
            color=ACCENT,
            lw=2.4,
            zorder=3,
        )
        ax.text(
            (fix.start_time + fix.end_time) / 2,
            fix.centroid_px[0] + 95,
            f"{fix.duration * 1000:.0f} ms",
            ha="center",
            fontsize=8,
            color=ACCENT,
        )

    ax.annotate(
        "one fixation, split in two\nby the blink between them",
        xy=((blink_start + blink_end) / 2, 1290),
        xytext=((blink_start + blink_end) / 2 + 0.75, 760),
        fontsize=8.5,
        color=BAD,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": BAD, "lw": 1.0},
    )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("screen x (px)")
    ax.set_ylim(300, 1650)
    ax.legend(ncol=2, loc="upper left")
    ax.set_title(
        f"I-VT: {len(fixations)} fixations with the dropout mask, "
        f"{len(without_mask)} without it",
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUT / "04_fixations.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_filters()
    figure_selection()
    figure_safety()
    figure_fixations()
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
