"""Auto-detect and fix the VL53 ToF 0x28/0x29 remap state, plus soft-reset.

Left sensor is 0x28 (XSHUT tied high, renamed). Right sensor is 0x29
(XSHUT on pin 29, stays at the chip default).

Used by 0_troubleshoot.ipynb (the one-click buttons). Same steps as the
manual cells: detect with i2cdetect, then Fix A (wake XSHUT) or
Fix B (full re-remap). Lives in scripts/ next to soft_reset.sh.

CLI self-check:  python3 scripts/remap_utils.py
"""
import re
import subprocess
import time
from pathlib import Path

SUDO_PASSWORD = "jetson"  # default Jetson password; change if you changed it
SERVICE = "tof-i2c-switcher-simple.service"


def _sudo(*cmd):
    return subprocess.run(
        ["sudo", "-S", *cmd],
        input=SUDO_PASSWORD + "\n", capture_output=True, text=True,
    )


def _xshut(value):
    """gpioset --mode=exit $(gpiofind PQ.05)=<value>"""
    parts = subprocess.run(["gpiofind", "PQ.05"], capture_output=True, text=True).stdout.split()
    if len(parts) != 2:
        raise RuntimeError("gpiofind PQ.05 failed — is gpiod installed?")
    chip, line = parts  # e.g. "gpiochip0", "81"
    return _sudo("gpioset", "--mode=exit", chip, f"{line}={value}")


def detect():
    """Return (has28, has29, raw_i2cdetect_output). Left is 0x28, right is 0x29."""
    p = _sudo("i2cdetect", "-y", "-r", "1")
    if p.returncode != 0:
        raise RuntimeError(f"i2cdetect failed: {p.stderr.strip()}")
    found = set()
    for line in p.stdout.splitlines():
        if re.match(r"^[0-7]0:", line):      # grid rows look like '20: -- -- 28 29 ...'
            found.update(line[4:].split())   # skip the row label, keep cell values
    return "28" in found, "29" in found, p.stdout


def resolve(log=print):
    """Detect, apply the matching fix, verify. Returns True if 0x28 and 0x29 are both present."""
    has28, has29, raw = detect()
    log(raw)
    if has28 and has29:
        log("OK: both sensors present (left 0x28 + right 0x29). Nothing to do.")
        return True
    if not has28 and not has29:
        log("HARDWARE: neither address on the bus.")
        log("Check sensor power/wiring, reseat VIN, or reboot. No software fix.")
        return False

    if has28:  # CASE A: right sensor (XSHUT, pin 29) asleep
        log("CASE A: only 0x28 — waking right sensor (pin 29 HIGH, comes up at 0x29)...")
        steps = [
            ("stop service (free the pin)", lambda: _sudo("systemctl", "stop", SERVICE)),
            ("pin 29 LOW (XSHUT reset)", lambda: _xshut(0)),
            ("pin 29 HIGH (right sensor wakes at 0x29)", lambda: _xshut(1)),
        ]
    else:  # CASE B: both reset, colliding on 0x29 -> hold right sensor deaf, rename left, wake right
        log("CASE B: only 0x29 — both sensors reset; renaming left sensor to 0x28...")
        steps = [
            ("stop service (free the pin)", lambda: _sudo("systemctl", "stop", SERVICE)),
            ("poke pinmux", lambda: _sudo("busybox", "devmem", "0x2430068", "w", "0x8")),
            ("pin 29 LOW (right sensor deaf)", lambda: _xshut(0)),
            ("rename left 0x29 -> 0x28", lambda: _sudo("i2ctransfer", "-y", "1", "w2@0x29", "0x8A", "0x28")),
            ("pin 29 HIGH (right sensor wakes at 0x29)", lambda: _xshut(1)),
        ]
    # ponytail: service left stopped on purpose (user's call) — pin stays HIGH on
    # this hardware after gpioset exits, but nothing holds it until next boot.

    for name, fn in steps:
        r = fn()
        log(f"  [{'ok' if r.returncode == 0 else 'warn'}] {name}"
            + (f": {r.stderr.strip()}" if r.returncode != 0 else ""))

    has28, has29, raw = detect()
    log(raw)
    ok = has28 and has29
    log("DONE: both sensors on the bus." if ok else "NOT FIXED — likely wiring/power; see above.")
    return ok


def remap_button():
    """Display the TLDR 'Resolve!' button (ipywidgets, preinstalled on jetcard)."""
    import ipywidgets as widgets
    from IPython.display import display

    out = widgets.Output()
    btn = widgets.Button(description="Resolve!", button_style="warning",
                         icon="wrench", layout=widgets.Layout(width="140px"))

    def _on_click(_):
        with out:
            out.clear_output()
            try:
                resolve()
            except Exception as e:
                print(f"ERROR: {e}")

    btn.on_click(_on_click)
    display(widgets.VBox([btn, out]))


# ---------------------------------------------------------------------------
# Soft reset — closest thing to a reboot, without one.
#
# MUST NOT run the steps in this kernel. Stopping Jupyter kills the cgroup
# (KillMode=control-group), so any child of this notebook dies too — that is
# why "stop Jupyter" first left the display still running.
#
# Fix: hand the whole sequence to systemd-run (own cgroup), and stop Jupyter
# LAST inside soft_reset.sh.
# ---------------------------------------------------------------------------

SOFT_RESET_LOG = "/tmp/jetracer_soft_reset.log"
SOFT_RESET_UNIT = "jetracer-soft-reset"


def sensor_soft_reset(log=print):
    """Fix addresses (same as Fix Sensors), then pulse XSHUT. Does not stop Jupyter.

    The script remaps to left 0x28 + right 0x29, then reboots the right sensor.
    The left sensor's XSHUT is tied high, so the pulse does not power-cycle it.
    """
    script = Path(__file__).resolve().with_name("sensor_soft_reset.sh")
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    log("Fixing sensor addresses, then resetting. This page stays open.")
    r = _sudo("bash", str(script))
    text = ((r.stdout or "") + (r.stderr or "")).strip()
    log(text or "(no output)")
    if r.returncode != 0:
        log("ERROR: sensor reset did not finish.")
        return False
    return True


def soft_reset(log=print):
    """Kick off the reset in a systemd job that survives Jupyter dying."""
    script = Path(__file__).resolve().with_name("soft_reset.sh")
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")

    log("Starting Soft Restart in the background.")
    log("Display, camera, and cache run FIRST. Jupyter is stopped LAST.")
    log("This page will disconnect in a few seconds — that is expected.")
    log("Wait ~15 seconds, then refresh the browser.")
    log(f"If it fails, on the Jetson run:  cat {SOFT_RESET_LOG}")

    # drop a leftover unit from a previous click so systemd-run can reuse the name
    _sudo("systemctl", "reset-failed", SOFT_RESET_UNIT)

    r = _sudo(
        "systemd-run",
        "--no-block",
        "--collect",
        f"--unit={SOFT_RESET_UNIT}",
        f"--property=StandardOutput=file:{SOFT_RESET_LOG}",
        f"--property=StandardError=file:{SOFT_RESET_LOG}",
        "/bin/bash", str(script),
    )
    if r.returncode != 0:
        log(f"ERROR: could not start reset job: {r.stderr.strip() or r.stdout.strip()}")
        return False
    log("  [ok] reset job started (systemd-run). You can wait for the disconnect.")
    return True


def _lock_buttons(buttons, busy):
    """Grey every button and drop clicks until unlocked."""
    busy["on"] = True
    for b in buttons:
        b.disabled = True


def _unlock_buttons_later(buttons, busy, seconds=5):
    """Re-enable after `seconds`, on the notebook UI thread when one exists."""
    import threading

    def _open():
        busy["on"] = False
        for b in buttons:
            b.disabled = False

    def _kick():
        try:
            from IPython import get_ipython
            get_ipython().kernel.io_loop.add_callback(_open)
        except Exception:
            _open()

    timer = threading.Timer(seconds, _kick)
    timer.daemon = True
    timer.start()


def _selfcheck_button_lock():
    class _Btn:
        disabled = False

    btn = _Btn()
    busy = {"on": False}
    _lock_buttons([btn], busy)
    assert btn.disabled and busy["on"]
    _unlock_buttons_later([btn], busy, 0.05)
    time.sleep(0.3)
    assert not btn.disabled and not busy["on"]


def troubleshoot_panel():
    """Buttons: fix addresses / reset a wedged sensor / soft-restart the car.

    Soft Restart only *starts* a systemd job, then this kernel is free
    to die. The job (not this notebook) stops Jupyter last.
    """
    import ipywidgets as widgets
    from IPython.display import display

    out = widgets.Output()
    buttons = []
    busy = {"on": False}

    def run(fn):
        def _on_click(_):
            if busy["on"]:
                return
            _lock_buttons(buttons, busy)
            started = time.monotonic()
            try:
                with out:
                    out.clear_output()
                    try:
                        fn()
                    except Exception as e:
                        print(f"ERROR: {e}")
            finally:
                # ponytail: 5s from the click; stay grey longer only if the action is still running
                _unlock_buttons_later(buttons, busy, max(0, 5 - (time.monotonic() - started)))
        return _on_click

    fix_btn = widgets.Button(description="Fix Sensors", button_style="warning",
                             icon="wrench",
                             layout=widgets.Layout(width="160px"))
    sensor_btn = widgets.Button(description="Reset Sensors", button_style="info",
                                icon="refresh",
                                layout=widgets.Layout(width="170px"))
    reset_btn = widgets.Button(description="Soft Restart", button_style="danger",
                               icon="refresh",
                               layout=widgets.Layout(width="170px"))
    both_btn = widgets.Button(description="Fix Everything", button_style="success",
                              icon="medkit",
                              layout=widgets.Layout(width="160px"))
    buttons.extend([fix_btn, sensor_btn, reset_btn, both_btn])

    def _both():
        print("== 1/2 fix sensors, then reset ==")
        sensor_soft_reset()
        print("== 2/2 car restart ==")
        soft_reset()

    fix_btn.on_click(run(resolve))
    sensor_btn.on_click(run(sensor_soft_reset))
    reset_btn.on_click(run(soft_reset))
    both_btn.on_click(run(_both))

    display(widgets.VBox([
        widgets.HTML(
            "<b>Car Troubleshooter</b><br>"
            "• <b>Fix Sensors</b> — distance sensors missing / wrong address, car still drives<br>"
            "• <b>Reset Sensors</b> — fix addresses, then reboot the right sensor "
            "(0x29). This page stays open<br>"
            "• <b>Soft Restart</b> — slow, out of memory, camera stuck "
            "(Jupyter restarts: this page will disconnect — just reopen it)<br>"
            "• <b>Fix Everything</b> — fix addresses, reset the sensors, then restart the car"
        ),
        widgets.HBox([fix_btn, sensor_btn, reset_btn, both_btn]),
        out,
    ]))


if __name__ == "__main__":
    resolve()
