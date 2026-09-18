"""Dwell against look-and-confirm, on identical gaze traces.

The two techniques are replayed through the same synthetic sessions, so any
difference between them is a difference between the techniques rather than
between two separately written event loops.

The session deliberately contains *reading* as well as *selecting*: the
simulated user spends time resting their gaze on targets they have no intention
of choosing. That is the situation dwell cannot distinguish from a selection —
the Midas touch — and it is the reason the numbers below come out as they do.

This is a simulation with a scripted user, not a user study. It shows what the
techniques do to a given trace; it says nothing about what people prefer, which
is what the questionnaires in `gazectl.study` are for.
"""

import numpy as np

from gazectl.selection import (
    DwellSelector,
    LookAndConfirmSelector,
    SelectionEvent,
    Target,
)

RATE_HZ = 60.0
SEED = 20260918
NOISE_PX = 8.0

TARGETS = [
    Target(id="left", x=200.0, y=400.0, width=260.0, height=200.0),
    Target(id="middle", x=830.0, y=400.0, width=260.0, height=200.0),
    Target(id="right", x=1460.0, y=400.0, width=260.0, height=200.0),
]
ELSEWHERE = np.array([960.0, 100.0])

#: (target id or None, seconds, whether the user means to select it).
#: "read" entries are the crux: the gaze rests on a target with no intent.
SCRIPT = [
    ("middle", 1.2, True),
    (None, 0.4, False),
    ("left", 1.0, False),  # reading, not choosing
    (None, 0.3, False),
    ("right", 1.2, True),
    (None, 0.4, False),
    ("middle", 0.9, False),  # reading again
    (None, 0.3, False),
    ("left", 1.2, True),
    (None, 0.5, False),
]

CONFIRM_DELAY_S = 0.25  # how long after arriving the user presses the key


def build_session(rng):
    """Expand the script into a noisy per-sample gaze trace."""
    points, times, intents = [], [], []
    t = 0.0

    for target_id, duration, intended in SCRIPT:
        centre = (
            next(x for x in TARGETS if x.id == target_id).centre_px if target_id else ELSEWHERE
        )
        n = int(duration * RATE_HZ)
        for _ in range(n):
            points.append(centre + rng.normal(0.0, NOISE_PX, size=2))
            times.append(t)
            intents.append((target_id, intended))
            t += 1.0 / RATE_HZ

    return np.array(points), np.array(times), intents


def intended_selections():
    return [target_id for target_id, _, intended in SCRIPT if intended]


def run_dwell(points, times, dwell_s):
    sel = DwellSelector(TARGETS, dwell_s=dwell_s, refractory_s=0.6)
    fired = []
    for i in range(len(points)):
        result = sel.update(points[i], times[i])
        if result.event is SelectionEvent.SELECTED:
            fired.append((result.target_id, times[i]))
    return fired


def run_look_and_confirm(points, times, intents):
    """The scripted user presses confirm only on segments they meant to select."""
    sel = LookAndConfirmSelector(TARGETS, confirm_window_s=0.3, min_fixation_s=0.15)
    fired = []

    pressed_for_segment = False
    previous_segment = None
    segment_start = times[0]

    for i in range(len(points)):
        target_id, intended = intents[i]
        segment = (target_id, intended)

        if segment != previous_segment:
            previous_segment = segment
            segment_start = times[i]
            pressed_for_segment = False

        sel.update(points[i], times[i])

        if intended and not pressed_for_segment and times[i] - segment_start >= CONFIRM_DELAY_S:
            result = sel.confirm(times[i])
            pressed_for_segment = True
            if result.event is SelectionEvent.SELECTED:
                fired.append((result.target_id, times[i]))

    return fired


def score(fired, wanted):
    """Count how the fired selections line up with what the user intended.

    Matching is by identity and order, which is the right notion here because
    the script never asks for the same target twice in a row.
    """
    fired_ids = [target_id for target_id, _ in fired]

    remaining = list(wanted)
    correct = 0
    for target_id in fired_ids:
        if target_id in remaining:
            remaining.remove(target_id)
            correct += 1

    return {
        "fired": len(fired_ids),
        "correct": correct,
        "unintended": len(fired_ids) - correct,
        "missed": len(remaining),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    points, times, intents = build_session(rng)
    wanted = intended_selections()

    print("Scripted session: three intended selections, two stretches of reading")
    print(f"  duration               {times[-1]:.1f} s")
    print(f"  intended selections    {len(wanted)}  ({', '.join(wanted)})")
    print()

    print(f"{'technique':<34} {'fired':>7} {'correct':>9} {'unintended':>12} {'missed':>8}")
    print("-" * 74)

    for dwell_s in (0.5, 0.8, 1.2):
        result = score(run_dwell(points, times, dwell_s), wanted)
        print(
            f"{'dwell ' + format(dwell_s, '.1f') + ' s':<34} {result['fired']:>7} "
            f"{result['correct']:>9} {result['unintended']:>12} {result['missed']:>8}"
        )

    result = score(run_look_and_confirm(points, times, intents), wanted)
    print(
        f"{'look-and-confirm':<34} {result['fired']:>7} "
        f"{result['correct']:>9} {result['unintended']:>12} {result['missed']:>8}"
    )

    print()
    print("Dwell has no setting that is right. A short dwell fires on the targets")
    print("the user was only reading; a long one starts missing the ones they")
    print("meant. Look-and-confirm has no unintended selections on this trace")
    print("because looking is not selecting -- which is exactly the property it")
    print("buys, and it pays for it by needing a hand.")


if __name__ == "__main__":
    main()
