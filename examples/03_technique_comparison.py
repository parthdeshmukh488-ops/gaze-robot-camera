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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Span:
    """One scripted segment of the session."""

    target_id: str | None
    intended: bool
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


def build_session(rng):
    """Expand the script into a noisy per-sample gaze trace.

    Returns the spans as well as the samples. Scoring needs to know *when*
    each scripted segment ran, not just which targets were wanted.
    """
    points, times, intents, spans = [], [], [], []
    t = 0.0

    for target_id, duration, intended in SCRIPT:
        centre = (
            next(x for x in TARGETS if x.id == target_id).centre_px if target_id else ELSEWHERE
        )
        start = t
        n = int(duration * RATE_HZ)
        for _ in range(n):
            points.append(centre + rng.normal(0.0, NOISE_PX, size=2))
            times.append(t)
            intents.append((target_id, intended))
            t += 1.0 / RATE_HZ
        spans.append(Span(target_id, intended, start, t))

    return np.array(points), np.array(times), intents, spans


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


def classify(fired, spans):
    """Label each fired selection by the span it landed in.

    Matching on target id alone is not good enough, and gets the attribution
    exactly backwards: a dwell that fires while the user is merely *reading*
    "left" would be matched against the later span where they genuinely wanted
    "left", scored as correct, and the real one then scored as a false
    positive. The totals happen to come out the same; the story they tell does
    not.

    A selection is correct when it fires inside a span that was intended, and
    names that span's target.
    """
    labelled = []
    satisfied = set()

    for target_id, when in fired:
        span = next((s for s in spans if s.contains(when)), None)
        correct = (
            span is not None
            and span.intended
            and span.target_id == target_id
            and span.start not in satisfied
        )
        if correct:
            satisfied.add(span.start)
        labelled.append((target_id, when, correct))

    wanted = [s for s in spans if s.intended]
    missed = sum(1 for s in wanted if s.start not in satisfied)

    return labelled, missed


def score(fired, spans):
    """Summary counts for one technique."""
    labelled, missed = classify(fired, spans)
    correct = sum(1 for _, _, ok in labelled if ok)

    return {
        "fired": len(labelled),
        "correct": correct,
        "unintended": len(labelled) - correct,
        "missed": missed,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    points, times, intents, spans = build_session(rng)
    wanted = [s.target_id for s in spans if s.intended]

    print("Scripted session: three intended selections, two stretches of reading")
    print(f"  duration               {times[-1]:.1f} s")
    print(f"  intended selections    {len(wanted)}  ({', '.join(wanted)})")
    print()

    print(f"{'technique':<34} {'fired':>7} {'correct':>9} {'unintended':>12} {'missed':>8}")
    print("-" * 74)

    for dwell_s in (0.5, 0.8, 1.2):
        result = score(run_dwell(points, times, dwell_s), spans)
        print(
            f"{'dwell ' + format(dwell_s, '.1f') + ' s':<34} {result['fired']:>7} "
            f"{result['correct']:>9} {result['unintended']:>12} {result['missed']:>8}"
        )

    result = score(run_look_and_confirm(points, times, intents), spans)
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
