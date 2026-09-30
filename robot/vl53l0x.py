import subprocess
import threading
import time
from pathlib import Path

# VL53L0X writes 8190/8191 when it has no target. Teaching notebooks show that as 2000.
TOF_CAP_MM = 2000
# Same gap the test script uses between samples, shortened from 0.1 s.
RANGE_PAUSE_S = 0.001
# Orin Nano JP6: bus 1 is pins 27/28, bus 7 is pins 3/5. Same map as test_two_sensors.py.
BLINKA_PINS = {
    1: ("SCL_1", "SDA_1"),
    7: ("SCL", "SDA"),
}
# 3 failed reads in a row, then at most one reset per 20s.
# ponytail: a real (2000, 2000) is "nothing in range", so it must not reset the
# car on an empty track. Only a failed read counts. A chip that still answers
# but is stuck at 2000 needs a separate stuck-streak check.
AUTO_RESET_STREAK = 3
AUTO_RESET_COOLDOWN_S = 20


def cap_mm(value, cap=TOF_CAP_MM):
    """Clamp a raw range. 8191 and anything above `cap` become `cap`."""
    value = int(value)
    if value <= 0 or value > cap:
        return cap
    return value


def hold_glitch(state, mm):
    """Pass a normal range through on the same frame.

    20 mm and 2000 mm are the flicker values. One of them keeps the previous
    reading. The same one more than three times in a row is accepted.
    """
    mm = int(mm)
    if mm not in (20, TOF_CAP_MM):
        state["good"] = mm
        state["n"] = 0
        return mm
    state["n"] = state["n"] + 1 if state.get("which") == mm else 1
    state["which"] = mm
    if state["n"] > 3:
        state["good"] = mm
        return mm
    return state["good"] if state.get("good") is not None else mm


def should_auto_reset(fail_streak, resetting, now, last_reset,
                      need=AUTO_RESET_STREAK, cooldown_s=AUTO_RESET_COOLDOWN_S):
    """True when `need` failed reads have piled up and the cooldown has elapsed."""
    if resetting or fail_streak < need:
        return False
    if last_reset and (now - last_reset) < cooldown_s:
        return False
    return True


def _sudo_password():
    try:
        from scripts.remap_utils import SUDO_PASSWORD
        return SUDO_PASSWORD
    except Exception:
        return "jetson"  # same lab default as scripts/remap_utils.py


def _run_hw_reset():
    """Fix addresses, then XSHUT-pulse. A failed 0x28 is software-reset over I2C."""
    script = Path(__file__).resolve().parents[1] / "JetRacer" / "scripts" / "sensor_soft_reset.sh"
    if not script.is_file():
        return False, "missing %s" % script
    r = subprocess.run(
        ["sudo", "-S", "bash", str(script)],
        input=_sudo_password() + "\n",
        capture_output=True, text=True,
    )
    text = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0, text


def _open_i2c(bus_id):
    """One shared bus, same Blinka pins as test_two_sensors.py."""
    import board
    import busio

    names = BLINKA_PINS.get(int(bus_id))
    if names is None:
        known = ", ".join(str(k) for k in sorted(BLINKA_PINS))
        raise RuntimeError("no Blinka pin map for i2c bus %s (known: %s)" % (bus_id, known))
    scl, sda = getattr(board, names[0]), getattr(board, names[1])
    return busio.I2C(scl, sda)


def _new_glitch():
    return {"good": None, "n": 0, "which": None}


class VL53Pair:
    """Left / right VL53L0X on one I2C bus.

    Both chips ship as 0x29. The left one is renamed to 0x28 (XSHUT tied
    high). The right one stays 0x29 (XSHUT on pin 29).

    ``auto_resolve=True`` runs ``sensor_soft_reset.sh`` once when open fails
    (timeout or any other error), then opens again. Later read failures still
    reset in the background after a short streak. A real 2000 mm reading does not.
    """

    def __init__(self, bus=1, addr_left=0x28, addr_right=0x29, auto_resolve=True) -> None:
        self._bus_num = bus
        self._addr_left = addr_left
        self._addr_right = addr_right
        self._auto_resolve = bool(auto_resolve)
        self._io_lock = threading.Lock()
        self._flag_lock = threading.Lock()
        self._resetting = False
        self._fail_streak = 0
        self._last_reset = 0.0
        self._i2c = None
        self.left = None
        self.right = None
        self._glitch_left = _new_glitch()
        self._glitch_right = _new_glitch()
        try:
            self._open()
        except Exception:
            if not self._auto_resolve:
                raise
            print("ToF open failed. Resetting the sensors.")
            _ok, text = _run_hw_reset()
            try:
                self._open()
            except Exception:
                if text:
                    print("\n".join(text.splitlines()[-6:]))
                raise
            print("ToF sensors are answering again after reset.")

    def _open(self) -> None:
        import adafruit_vl53l0x

        self._i2c = _open_i2c(self._bus_num)
        try:
            # Same constructor the test script uses. io_timeout_s stops a dead
            # chip from sitting in .range forever; the test script leaves it at 0.
            self.left = adafruit_vl53l0x.VL53L0X(
                self._i2c, address=self._addr_left, io_timeout_s=0.1)
            self.right = adafruit_vl53l0x.VL53L0X(
                self._i2c, address=self._addr_right, io_timeout_s=0.1)
        except Exception:
            self._close()
            raise
        self._glitch_left = _new_glitch()
        self._glitch_right = _new_glitch()

    def _close(self) -> None:
        self.left = None
        self.right = None
        i2c = self._i2c
        self._i2c = None
        if i2c is not None:
            try:
                i2c.deinit()
            except Exception:
                pass

    def _clear_glitch(self) -> None:
        self._glitch_left.update(good=None, n=0, which=None)
        self._glitch_right.update(good=None, n=0, which=None)

    def _sample(self, sensor, state) -> int:
        """adafruit ``.range``, then 0.001 s. Cap and glitch filter stay for the notebooks."""
        mm = int(sensor.range)
        time.sleep(RANGE_PAUSE_S)
        return hold_glitch(state, cap_mm(mm))

    def read_mm(self):
        """(left_mm, right_mm).

        A failed read (any error, not only OSError) returns (2000, 2000) and,
        after a short streak, resets the sensors on a background thread.
        A real reading of 2000 mm does not. The caller is not blocked while
        a reset is in progress.
        """
        if self._resetting or not self._io_lock.acquire(blocking=False):
            return (TOF_CAP_MM, TOF_CAP_MM)
        try:
            if self.left is not None and self.right is not None:
                try:
                    sample = (
                        self._sample(self.left, self._glitch_left),
                        self._sample(self.right, self._glitch_right),
                    )
                except Exception:
                    self._clear_glitch()
                else:
                    self._fail_streak = 0
                    return sample
        finally:
            self._io_lock.release()
        self._fail_streak += 1
        self._maybe_auto_reset()
        return (TOF_CAP_MM, TOF_CAP_MM)

    def _maybe_auto_reset(self) -> None:
        if not self._auto_resolve:
            return
        if should_auto_reset(self._fail_streak, self._resetting, time.time(), self._last_reset):
            self.request_reset(blocking=False, reason="ToF read failed. Resetting the sensors.")

    def request_reset(self, blocking=False, reason="Resetting the sensors.") -> None:
        """Close the bus, pulse XSHUT, and initialise the chips again.

        `blocking=False` is for the read loop. The notebook button uses
        `blocking=True` on its own thread. Either way `read_mm` keeps
        returning (2000, 2000) until the sensors answer.
        """
        with self._flag_lock:
            if self._resetting:
                return
            self._resetting = True
            self._last_reset = time.time()
        print(reason)
        if blocking:
            self._recover()
        else:
            threading.Thread(target=self._recover, daemon=True).start()

    def _reopen(self) -> bool:
        with self._io_lock:
            self._close()
            try:
                self._open()
                self._sample(self.left, self._glitch_left)
                self._sample(self.right, self._glitch_right)
            except Exception:
                self._close()
                return False
            self._fail_streak = 0
            return True

    def _recover(self) -> None:
        try:
            if self._reopen():
                print("ToF sensors are answering again.")
                return
            _ok, text = _run_hw_reset()
            if self._reopen():
                print("ToF sensors are answering again after reset.")
                return
            print("ToF reset did not bring the sensors back.")
            if text:
                print("\n".join(text.splitlines()[-6:]))
        finally:
            self._resetting = False


if __name__ == "__main__":
    assert cap_mm(8191) == 2000 and cap_mm(20) == 20 and cap_mm(221) == 221
    state = {"good": None, "n": 0, "which": None}
    got = [hold_glitch(state, v) for v in (245, 20, 267, 2000, 250)]
    assert got == [245, 245, 267, 267, 250], got
    state = {"good": None, "n": 0, "which": None}
    got = [hold_glitch(state, v) for v in (245, 2000, 2000, 2000, 2000)]
    assert got == [245, 245, 245, 245, 2000], got
    state = {"good": None, "n": 0, "which": None}
    got = [hold_glitch(state, v) for v in (245, 20, 20, 20, 20)]
    assert got == [245, 245, 245, 245, 20], got
    assert should_auto_reset(2, False, 100, 0) is False
    assert should_auto_reset(3, False, 100, 0) is True
    assert should_auto_reset(5, True, 100, 0) is False
    assert should_auto_reset(5, False, 110, 100) is False
    assert should_auto_reset(5, False, 130, 100) is True

    class _Pair:
        _auto_resolve = False
        _fail_streak = 5
        _resetting = False
        _last_reset = 0.0
        calls = []

        def request_reset(self, blocking=False, reason=""):
            self.calls.append(reason)

    quiet = _Pair()
    VL53Pair._maybe_auto_reset(quiet)
    assert quiet.calls == []
    quiet._auto_resolve = True
    VL53Pair._maybe_auto_reset(quiet)
    assert quiet.calls == ["ToF read failed. Resetting the sensors."]
    print("vl53l0x filter ok")
