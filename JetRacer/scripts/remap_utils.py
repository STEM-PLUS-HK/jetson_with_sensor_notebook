"""Auto-detect and fix the VL53 ToF 0x29/0x30 remap state, plus soft-reset.

Used by 0_troubleshoot.ipynb (the one-click buttons). Same steps as the
manual cells: detect with i2cdetect, then Fix A (wake XSHUT) or
Fix B (full re-remap). Lives in scripts/ next to soft_reset.sh.

CLI self-check:  python3 scripts/remap_utils.py
"""
import re
import subprocess
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
    """Return (has29, has30, raw_i2cdetect_output)."""
    p = _sudo("i2cdetect", "-y", "-r", "1")
    if p.returncode != 0:
        raise RuntimeError(f"i2cdetect failed: {p.stderr.strip()}")
    found = set()
    for line in p.stdout.splitlines():
        if re.match(r"^[0-7]0:", line):      # grid rows look like '30: -- -- 33 ...'
            found.update(line[4:].split())   # skip the row label, keep cell values
    return "29" in found, "30" in found, p.stdout


def resolve(log=print):
    """Detect, apply the matching fix, verify. Returns True if 29+30 both present."""
    has29, has30, raw = detect()
    log(raw)
    if has29 and has30:
        log("OK: both sensors present (0x29 + 0x30). Nothing to do.")
        return True
    if not has29 and not has30:
        log("HARDWARE: neither address on the bus.")
        log("Check sensor power/wiring, reseat VIN, or reboot. No software fix.")
        return False

    if has30:  # CASE A: XSHUT sensor asleep -> pulse pin 29 LOW then HIGH
        log("CASE A: only 0x30 — waking XSHUT sensor (pin 29 HIGH)...")
        steps = [
            ("stop service (free the pin)", lambda: _sudo("systemctl", "stop", SERVICE)),
            ("pin 29 LOW (XSHUT reset)", lambda: _xshut(0)),
            ("pin 29 HIGH (XSHUT wakes at 0x29)", lambda: _xshut(1)),
        ]
    else:  # CASE B: both reset, colliding on 0x29 -> low, rename, high
        log("CASE B: only 0x29 — both sensors reset; re-remapping...")
        steps = [
            ("stop service (free the pin)", lambda: _sudo("systemctl", "stop", SERVICE)),
            ("poke pinmux", lambda: _sudo("busybox", "devmem", "0x2430068", "w", "0x8")),
            ("pin 29 LOW (XSHUT deaf)", lambda: _xshut(0)),
            ("rename 0x29 -> 0x30", lambda: _sudo("i2ctransfer", "-y", "1", "w2@0x29", "0x8A", "0x30")),
            ("pin 29 HIGH (XSHUT wakes at 0x29)", lambda: _xshut(1)),
        ]
    # ponytail: service left stopped on purpose (user's call) — pin stays HIGH on
    # this hardware after gpioset exits, but nothing holds it until next boot.

    for name, fn in steps:
        r = fn()
        log(f"  [{'ok' if r.returncode == 0 else 'warn'}] {name}"
            + (f": {r.stderr.strip()}" if r.returncode != 0 else ""))

    has29, has30, raw = detect()
    log(raw)
    ok = has29 and has30
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


def troubleshoot_panel():
    """Three buttons for students: fix sensors / soft reset / both.

    Soft Restart only *starts* a systemd job, then this kernel is free
    to die. The job (not this notebook) stops Jupyter last.
    """
    import ipywidgets as widgets
    from IPython.display import display

    out = widgets.Output()

    def run(fn):
        def _on_click(_):
            with out:
                out.clear_output()
                try:
                    fn()
                except Exception as e:
                    print(f"ERROR: {e}")
        return _on_click

    fix_btn = widgets.Button(description="Fix Sensors", button_style="warning",
                             icon="wrench",
                             layout=widgets.Layout(width="170px"))
    reset_btn = widgets.Button(description="Soft Restart", button_style="danger",
                               icon="refresh",
                               layout=widgets.Layout(width="190px"))
    both_btn = widgets.Button(description="Fix Everything", button_style="success",
                              icon="medkit",
                              layout=widgets.Layout(width="170px"))

    fix_btn.on_click(run(resolve))
    reset_btn.on_click(run(soft_reset))

    def _both(_):
        with out:
            out.clear_output()
            try:
                print("== 1/2 sensors ==")
                resolve()
                print("== 2/2 soft reset ==")
                soft_reset()
            except Exception as e:
                print(f"ERROR: {e}")
    both_btn.on_click(_both)

    display(widgets.VBox([
        widgets.HTML(
            "<b>Car Troubleshooter</b><br>"
            "• <b>Fix Sensors</b> — distance sensors missing / wrong address, car still drives<br>"
            "• <b>Soft Restart</b> — slow, out of memory, camera stuck "
            "(Jupyter restarts: this page will disconnect — just reopen it)<br>"
            "• <b>Fix Everything</b> — sensors first, then the restart"
        ),
        widgets.HBox([fix_btn, reset_btn, both_btn]),
        out,
    ]))


if __name__ == "__main__":
    resolve()
