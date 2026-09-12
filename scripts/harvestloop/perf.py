"""
Wall-clock speed controls and per-stage timing.

Nothing here changes what the simulation computes. The route runs the
same number of control steps with the same physics dt and the same
tolerances; these knobs only stop the app from waiting around between
those steps, and stop it from rendering an image nobody reads.

Two costs dominate a route run:

    1. The Kit editor rate-limits its main loop to 60 Hz, so every
       control step waits for the next 60 Hz tick even when the step
       itself took a fraction of that.

    2. The persistent wrist render product renders every frame for
       the whole route, while only a couple of frames per truss are
       ever saved.
"""

import time

import carb.settings

from .config import *  # noqa: F401,F403


# Run loops that are rate limited by the app. The main loop is the one
# the route awaits; the others are throttled alongside it.
_RATE_LIMIT_SETTINGS = (
    "/app/runLoops/main/rateLimitEnabled",
    "/app/runLoops/present/rateLimitEnabled",
    "/app/runLoops/rendering_0/rateLimitEnabled",
    "/app/runLoops/rendering_1/rateLimitEnabled",
)

# What makes removing the rate limit safe. In manual mode with fixed
# time stepping, an app update advances simulation time by exactly
# 1 / rateLimitFrequency no matter how long the update took in real
# time, so the route sees the same dt per control step either way.
# Without them, dt would follow the wall clock and a faster loop would
# quietly change the dynamics the controllers were tuned against.
_FIXED_STEP_SETTINGS = (
    "/app/runLoops/main/manualModeEnabled",
    "/app/player/useFixedTimeStepping",
)


class RunLoopRateLimit:
    """
    Lets the app step as fast as it can compute, then restores it.

    Same steps, same dt, same results; only the real time between
    steps changes. The editor UI becomes less responsive while the
    route runs, which is why the original values are restored.
    """

    def __init__(self):
        self.saved = {}

    def disable(self):
        if not UNTHROTTLE_RUN_LOOP:
            return

        settings = carb.settings.get_settings()

        print()
        print("========================================")
        print("[PERF] RUN LOOP UNTHROTTLED")
        print("========================================")

        # Pin fixed-dt stepping first, so no frame can run on a
        # wall-clock dt once the throttle comes off.
        for path in _FIXED_STEP_SETTINGS:
            if settings.get(path):
                continue

            self.saved[path] = bool(
                settings.get(path)
            )

            settings.set_bool(
                path,
                True,
            )

            print(
                f"[PERF] Enabled {path} to keep dt fixed."
            )

        for path in _RATE_LIMIT_SETTINGS:
            current = settings.get(path)

            if current is None:
                continue

            self.saved[path] = bool(
                current
            )

            settings.set_bool(
                path,
                False,
            )

        frequency = settings.get(
            "/app/runLoops/main/rateLimitFrequency"
        )

        print(
            f"Rate limits disabled: {len(self.saved)} settings | "
            f"step dt still 1/{frequency} s"
        )

        print(
            "The editor stays busy until the route ends."
        )

    def restore(self):
        if not self.saved:
            return

        settings = carb.settings.get_settings()

        for path, value in self.saved.items():
            settings.set_bool(
                path,
                bool(value),
            )

        print(
            f"[PERF] Restored {len(self.saved)} run loop settings."
        )

        self.saved = {}


class ViewportUpdates:
    """
    Optionally stops the main viewport rendering during the route.

    The saved images come from the wrist render product, not from
    this viewport, so freezing it removes a per-frame render without
    touching a single measurement. It also removes the only way to
    watch the run, hence the config switch.
    """

    def __init__(self):
        self.viewport = None

    def pause(self):
        if not FREEZE_VIEWPORT_DURING_ROUTE:
            return

        try:
            from omni.kit.viewport.utility import (
                get_active_viewport,
            )

            viewport = get_active_viewport()

            if viewport is None:
                return

            viewport.updates_enabled = False
            self.viewport = viewport
        except Exception as exc:
            print(
                f"[PERF] Could not pause the viewport: {exc}"
            )
            return

        print(
            "[PERF] Viewport frozen for the run "
            "(wrist captures are unaffected)."
        )

    def resume(self):
        if self.viewport is None:
            return

        try:
            self.viewport.updates_enabled = True

            print(
                "[PERF] Viewport rendering restored."
            )
        except Exception as exc:
            print(
                f"[PERF] Could not restore the viewport: {exc}"
            )

        self.viewport = None


def set_render_product_updates(
    capture_session,
    enabled,
):
    """
    Turn the wrist render product on only while capturing.

    The render product created for the wrist camera otherwise renders
    on every app update for the entire route, which costs a lot of
    frames for two saved images per truss.

    Returns True when the toggle actually applied.
    """
    if (
        not GATE_WRIST_RENDER_PRODUCT
        or capture_session is None
    ):
        return False

    render_product = capture_session.get(
        "render_product"
    )

    hydra_texture = getattr(
        render_product,
        "hydra_texture",
        None,
    )

    if hydra_texture is None:
        return False

    try:
        hydra_texture.set_updates_enabled(
            bool(enabled)
        )
    except Exception as exc:
        # Never let a performance knob break a run.
        print(
            f"[PERF] Could not toggle wrist render product: {exc}"
        )
        return False

    return True


class StageTimer:
    """
    Wall-clock seconds per named stage, for one route run.

    Recorded so a speed change can be checked against the previous
    run's outcomes rather than assumed: if the per-stage totals drop
    while the per-truss errors do not move, nothing was traded away.
    """

    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.started_at = time.perf_counter()

    def add(
        self,
        stage,
        seconds,
    ):
        self.totals[stage] = (
            self.totals.get(stage, 0.0)
            + float(seconds)
        )

        self.counts[stage] = (
            self.counts.get(stage, 0)
            + 1
        )

    def measure(
        self,
        stage,
    ):
        """Context manager: `with timer.measure("GRASP"): ...`"""
        return _Measurement(
            self,
            stage,
        )

    def elapsed(self):
        return (
            time.perf_counter()
            - self.started_at
        )

    def summary_lines(self):
        lines = [
            "========================================",
            "[STAGE TIMING]",
            "========================================",
        ]

        total = self.elapsed()

        lines.append(
            f"Route wall clock: {total / 60.0:.1f} min"
        )

        if not self.totals:
            return lines

        lines.append(
            ""
        )

        lines.append(
            f"{'stage':<22}{'total_s':>9}"
            f"{'calls':>7}{'mean_s':>9}{'share':>8}"
        )

        for stage, seconds in sorted(
            self.totals.items(),
            key=lambda item: -item[1],
        ):
            calls = self.counts[stage]

            lines.append(
                f"{stage:<22}{seconds:>9.1f}{calls:>7d}"
                f"{seconds / max(calls, 1):>9.2f}"
                f"{100.0 * seconds / max(total, 1e-9):>7.1f}%"
            )

        return lines

    def print_summary(self):
        print()

        for line in self.summary_lines():
            print(
                line
            )


def measure(
    timing,
    stage,
):
    """
    `with measure(timing, "GRASP"): ...`, tolerating timing=None.

    Lets the harvest code be instrumented without every call site
    having to check whether a timer was passed in.
    """
    if timing is None:
        return _NullMeasurement()

    return timing.measure(
        stage
    )


class _NullMeasurement:
    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False


class _Measurement:
    def __init__(
        self,
        timer,
        stage,
    ):
        self.timer = timer
        self.stage = stage
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.timer.add(
            self.stage,
            time.perf_counter() - self.start,
        )

        return False
