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


def _reset_script_paths(root, cwd):
    """USB install writes the script under JetRacer/with_sensor/scripts/."""
    name = Path("scripts") / "sensor_soft_reset.sh"
    return (
        Path(root) / "JetRacer" / "with_sensor" / name,
        Path(cwd) / "with_sensor" / name,
        Path(cwd) / name,
        Path(root) / "JetRacer" / name,
    )


def _reset_script():
    for script in _reset_script_paths(Path(__file__).resolve().parents[1], Path.cwd()):
        if script.is_file():
            return script
    return None


def _run_hw_reset():
    """Fix addresses, then XSHUT-pulse. A failed 0x28 is software-reset over I2C."""
    script = _reset_script()
    if script is None:
        looked = _reset_script_paths(Path(__file__).resolve().parents[1], Path.cwd())
        return False, "missing sensor_soft_reset.sh\n" + "\n".join(str(p) for p in looked)
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


# One open pair per bus. Re-running the notebook cell must close the old reader
# or both use I2C at once (errno 121). The background thread keeps the old object alive.
_OPEN_PAIRS = {}


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
        self._closed = False
        self._latest = (TOF_CAP_MM, TOF_CAP_MM)
        self._sampled = threading.Event()
        self._range_stop = threading.Event()
        self._range_thread = None
        prev = _OPEN_PAIRS.get(self._bus_num)
        if prev is not None and prev is not self:
            prev.close()
            thread = prev._range_thread
            if thread is not None and thread.is_alive():
                raise RuntimeError(
                    "ToF reader is still on the bus. Restart the kernel and run this cell again.")
        _OPEN_PAIRS[self._bus_num] = self
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
        # The chips take ~0.07 s. Read them off to the side so the drive loop
        # can keep up with the camera. read_mm returns the newest pair.
        self._range_thread = threading.Thread(target=self._range_loop, daemon=True)
        self._range_thread.start()

    def close(self) -> None:
        """Stop the background read and release I2C so a new VL53Pair can open.

        Safe to call again. If the reader is still inside a measurement, the bus
        stays open until a later close, so a new pair does not share it.
        """
        self._closed = True
        self._range_stop.set()
        thread = self._range_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
            if thread.is_alive():
                return
        if self._i2c is None and self.left is None:
            if _OPEN_PAIRS.get(self._bus_num) is self:
                del _OPEN_PAIRS[self._bus_num]
            return
        if self._io_lock.acquire(timeout=1.5):
            try:
                self._close()
            finally:
                self._io_lock.release()
        if _OPEN_PAIRS.get(self._bus_num) is self and self._i2c is None:
            del _OPEN_PAIRS[self._bus_num]

    def _range_loop(self) -> None:
        while not self._range_stop.is_set() and not self._closed:
            if self._resetting:
                time.sleep(0.05)
                continue
            self._latest = self._read_blocking()
            self._sampled.set()

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
        """(left_mm, right_mm). Returns the newest pair and does not wait.

        A failed read (any error, not only OSError) returns (2000, 2000) and,
        after a short streak, resets the sensors on a background thread.
        A real reading of 2000 mm does not. The caller is not blocked while
        a reset is in progress.
        """
        if not self._sampled.is_set():
            self._sampled.wait(0.3)
        if self._resetting:
            return (TOF_CAP_MM, TOF_CAP_MM)
        return self._latest

    def _read_blocking(self):
        if self._closed or self._range_stop.is_set() or self._resetting or not self._io_lock.acquire(blocking=False):
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
        # A close or reset may have started during this measurement. Do not
        # launch another reset, or it will use I2C while the new pair opens.
        if self._closed or self._range_stop.is_set() or self._resetting:
            return (TOF_CAP_MM, TOF_CAP_MM)
        self._fail_streak += 1
        self._maybe_auto_reset()
        return (TOF_CAP_MM, TOF_CAP_MM)

    def _maybe_auto_reset(self) -> None:
        if self._closed or self._range_stop.is_set() or not self._auto_resolve:
            return
        if should_auto_reset(self._fail_streak, self._resetting, time.time(), self._last_reset):
            self.request_reset(blocking=False, reason="ToF read failed. Resetting the sensors.")

    def request_reset(self, blocking=False, reason="Resetting the sensors.") -> None:
        """Close the bus, pulse XSHUT, and initialise the chips again.

        `blocking=False` is for the read loop. Either way `read_mm` keeps
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

    def reset_sensors(self) -> None:
        """Run the same script as troubleshoot Reset Sensors, then open the chips.

        The bus is closed first so the script can use I2C. `read_mm` returns
        (2000, 2000) until the sensors answer.
        """
        from scripts.remap_utils import sensor_soft_reset

        with self._flag_lock:
            if self._resetting:
                return
            self._resetting = True
            self._last_reset = time.time()
        try:
            with self._io_lock:
                self._close()
            sensor_soft_reset()
            if self._reopen():
                print("ToF sensors are answering again after reset.")
            else:
                print("ToF reset did not bring the sensors back.")
        finally:
            self._resetting = False

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
        _closed = False
        _range_stop = threading.Event()
        calls = []

        def request_reset(self, blocking=False, reason=""):
            self.calls.append(reason)

    quiet = _Pair()
    VL53Pair._maybe_auto_reset(quiet)
    assert quiet.calls == []
    quiet._auto_resolve = True
    VL53Pair._maybe_auto_reset(quiet)
    assert quiet.calls == ["ToF read failed. Resetting the sensors."]
    cached = VL53Pair.__new__(VL53Pair)
    cached._resetting = False
    cached._latest = (11, 22)
    cached._sampled = threading.Event()
    cached._sampled.set()
    assert VL53Pair.read_mm(cached) == (11, 22)
    cached._resetting = True
    assert VL53Pair.read_mm(cached) == (2000, 2000)
    old = VL53Pair.__new__(VL53Pair)
    old._bus_num = 99
    old._closed = False
    old._range_stop = threading.Event()
    old._range_thread = None
    old._io_lock = threading.Lock()
    old._i2c = None
    old.left = None
    old.right = None
    _OPEN_PAIRS[99] = old
    old.close()
    assert old._closed and 99 not in _OPEN_PAIRS
    old.close()
    dead = VL53Pair.__new__(VL53Pair)
    dead._closed = True
    dead._range_stop = threading.Event()
    dead._range_stop.set()
    dead._resetting = False
    dead._io_lock = threading.Lock()
    dead._fail_streak = 0
    dead._auto_resolve = True
    dead._i2c = None
    dead.left = None
    dead.right = None
    assert VL53Pair._read_blocking(dead) == (2000, 2000)
    assert dead._fail_streak == 0
    paths = [str(p) for p in _reset_script_paths("/home/jetson/JetBot-JetRacer", "/home/jetson/JetBot-JetRacer/JetRacer")]
    assert paths[0].endswith("JetRacer/with_sensor/scripts/sensor_soft_reset.sh"), paths
    assert any(p.endswith("JetRacer/scripts/sensor_soft_reset.sh") for p in paths)
    print("vl53l0x filter ok")
