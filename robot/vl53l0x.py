from smbus import SMBus
import time


class VL53L0X:
    """ST VL53L0X ToF ranger over I2C (default address 0x29)."""

    SYSRANGE_START = 0x00
    RESULT_INTERRUPT_STATUS = 0x13
    RESULT_RANGE_STATUS = 0x14
    RESULT_RANGE_MM = 0x1E
    I2C_SLAVE_DEVICE_ADDRESS = 0x8A
    IDENTIFICATION_MODEL_ID = 0xC0

    def __init__(self, bus=1, address=0x29) -> None:
        self.bus = SMBus(bus)
        self.address = address
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
        if value <= 0 or value > 2000:
            return 2000
        return int(value)

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


class VL53Pair:
    """Left / right VL53L0X on one I2C bus.

    Both chips ship as 0x29. Change one address (often via XSHUT) so
    left stays 0x29 and right is 0x30, or pass two already-unique addresses.
    """

    def __init__(self, bus=1, addr_left=0x29, addr_right=0x30) -> None:
        self.left = VL53L0X(bus=bus, address=addr_left)
        self.right = VL53L0X(bus=bus, address=addr_right)

    def read_mm(self):
        return self.left.range_mm(), self.right.range_mm()
