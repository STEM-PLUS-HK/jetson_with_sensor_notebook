from smbus import SMBus
import subprocess
import threading
import time
from pathlib import Path

# VL53L0X writes 8190/8191 when it has no target. Teaching notebooks show that as 2000.
TOF_CAP_MM = 2000
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
    """Fix addresses, then XSHUT-pulse. The left sensor's XSHUT is tied high."""
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


class VL53L0X:
    """ST VL53L0X ToF ranger over I2C (default address 0x29)."""

    SYSRANGE_START = 0x00
    RESULT_INTERRUPT_STATUS = 0x13
    RESULT_RANGE_STATUS = 0x14
    RESULT_RANGE_MM = 0x1E
    I2C_SLAVE_DEVICE_ADDRESS = 0x8A
    IDENTIFICATION_MODEL_ID = 0xC0

    def __init__(self, bus=1, address=0x29) -> None:
        self.bus_num = bus
        self.bus = SMBus(bus)
        self.address = address
        self._glitch = {"good": None, "n": 0, "which": None}
        model_id = self._read_u8(self.IDENTIFICATION_MODEL_ID)
        if model_id != 0xEE:
            raise RuntimeError('VL53L0X not found at 0x%02X (id=0x%02X)' % (address, model_id))
        self._data_init()

    def set_address(self, new_addr) -> None:
        self._write_u8(self.I2C_SLAVE_DEVICE_ADDRESS, new_addr & 0x7F)
        self.address = new_addr

    def range_mm(self) -> int:
        self._write_u8(0x80, 0x01)
        self._write_u8(0xFF, 0x01)
        self._write_u8(0x00, 0x00)
        self._write_u8(0x91, self._stop_variable)
        self._write_u8(0x00, 0x01)
        self._write_u8(0xFF, 0x00)
        self._write_u8(0x80, 0x00)
        self._write_u8(self.SYSRANGE_START, 0x01)
        for _ in range(100):
            if (self._read_u8(self.SYSRANGE_START) & 0x01) == 0:
                break
            time.sleep(0.001)
        value = self._read_u16(self.RESULT_RANGE_MM)
        self._write_u8(0x0B, 0x01)
        return hold_glitch(self._glitch, cap_mm(value))

    def _data_init(self) -> None:
        self._write_u8(0x88, 0x00)
        self._write_u8(0x80, 0x01)
        self._write_u8(0xFF, 0x01)
        self._write_u8(0x00, 0x00)
        self._stop_variable = self._read_u8(0x91)
        self._write_u8(0x00, 0x01)
        self._write_u8(0xFF, 0x00)
        self._write_u8(0x80, 0x00)

    def _write_u8(self, reg, val) -> None:
        self.bus.write_byte_data(self.address, reg, val & 0xFF)

    def _read_u8(self, reg) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def _read_u16(self, reg) -> int:
        data = self.bus.read_i2c_block_data(self.address, reg, 2)
        return (data[0] << 8) | data[1]

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass


class VL53Pair:
    """Left / right VL53L0X on one I2C bus.

    Both chips ship as 0x29. The left one is renamed to 0x28 (XSHUT tied
    high). The right one stays 0x29 (XSHUT on pin 29).
    """

    def __init__(self, bus=1, addr_left=0x28, addr_right=0x29) -> None:
        self._bus_num = bus
        self._addr_left = addr_left
        self._addr_right = addr_right
        self._io_lock = threading.Lock()
        self._flag_lock = threading.Lock()
        self._resetting = False
        self._fail_streak = 0
        self._last_reset = 0.0
        self.left = None
        self.right = None
        self._open()

    def _open(self) -> None:
        self.left = VL53L0X(bus=self._bus_num, address=self._addr_left)
        try:
            self.right = VL53L0X(bus=self._bus_num, address=self._addr_right)
        except Exception:
            self.left.close()
            self.left = None
            raise

    def _close(self) -> None:
        for dev in (self.left, self.right):
            if dev is not None:
                dev.close()
        self.left = None
        self.right = None

    def _clear_glitch(self) -> None:
        for dev in (self.left, self.right):
            if dev is not None:
                dev._glitch.update(good=None, n=0, which=None)

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
                    sample = (self.left.range_mm(), self.right.range_mm())
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
                self.left.range_mm()
                self.right.range_mm()
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
    print("vl53l0x filter ok")
