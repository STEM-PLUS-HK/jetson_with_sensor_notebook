#!/usr/bin/env bash
# Fix sensor addresses (same cases as Fix Sensors), then pulse XSHUT.
# Does not stop Jupyter.
# Pin 29 is XSHUT for the right sensor (0x29). The left sensor (0x28)
# has XSHUT tied high, so the pulse cannot power-cycle it. If 0x28 then
# fails the model-id read, the script software-resets that chip over I2C.
set -u
echo "== $(date) sensor reset start =="
echo "== 1/2 fix sensor addresses =="

systemctl stop tof-i2c-switcher-simple.service || true

echo "pinmux"
busybox devmem 0x2430068 w 0x8 || echo "pinmux poke failed"

found=$(gpiofind PQ.05 || true)
chip=${found%% *}
rest=${found#* }
line=${rest%% *}
if [ -z "${chip}" ] || [ -z "${line}" ] || [ "${line}" = "${chip}" ]; then
    echo "gpiofind PQ.05 failed — is gpiod installed?"
    exit 1
fi

detect=$(i2cdetect -y -r 1 || true)
echo "$detect"
has28=0
has29=0
echo "$detect" | grep -qw 28 && has28=1
echo "$detect" | grep -qw 29 && has29=1

if [ "$has28" -eq 1 ] && [ "$has29" -eq 1 ]; then
    echo "OK: both sensors present (left 0x28 + right 0x29). Nothing to remap."
elif [ "$has28" -eq 0 ] && [ "$has29" -eq 0 ]; then
    echo "HARDWARE: neither address on the bus."
    echo "Check sensor power/wiring, reseat VIN, or reboot. Continuing with the XSHUT pulse."
elif [ "$has28" -eq 1 ]; then
    echo "CASE A: only 0x28 — waking right sensor (pin 29 HIGH, comes up at 0x29)..."
    gpioset --mode=exit "$chip" "${line}=0"
    gpioset --mode=exit "$chip" "${line}=1" || true
else
    echo "CASE B: only 0x29 — renaming left sensor to 0x28..."
    gpioset --mode=exit "$chip" "${line}=0"
    i2ctransfer -y 1 w2@0x29 0x8A 0x28 || echo "rename 0x29 -> 0x28 failed"
    gpioset --mode=exit "$chip" "${line}=1" || true
fi

echo "== 2/2 XSHUT reset =="
echo "XSHUT low for 1s (right sensor, 0x29, in reset)"
gpioset --mode=time --sec=1 "$chip" "${line}=0"
echo "XSHUT high (right sensor boots again at 0x29)"
gpioset --mode=exit "$chip" "${line}=1" || true
sleep 0.1

echo "bus after reset:"
i2cdetect -y -r 1 || true

echo "register read (i2cdetect only checks the address; this is the real test):"
probe() {
    local addr="$1" i out
    for i in 1 2 3 4 5; do
        if out=$(i2cget -y 1 "$addr" 0xC0 2>&1); then
            echo "  $addr model $out"
            return 0
        fi
        sleep 0.15
    done
    echo "  $addr FAILED: $out"
    return 1
}
left_ok=0
right_ok=0
probe 0x28 && left_ok=1
probe 0x29 && right_ok=1

# 0xBF is SOFT_RESET. The chip comes back at 0x29, so 0x29 is held in
# XSHUT first. A chip that never ACKs cannot be reset this way.
if [ "$left_ok" -eq 0 ]; then
    echo "== 0x28 failed. I2C software reset of the left sensor =="
    echo "Pin 29 held low so the right sensor is off the bus."
    gpioset --mode=exit "$chip" "${line}=0"
    sleep 0.05
    if i2cset -y 1 0x28 0xBF 0x00; then
        sleep 0.01
        i2cset -y 1 0x29 0xBF 0x01 || i2cset -y 1 0x28 0xBF 0x01 || echo "release soft reset failed"
        sleep 0.1
        if i2cget -y 1 0x29 0xC0 >/dev/null 2>&1; then
            i2ctransfer -y 1 w2@0x29 0x8A 0x28 || echo "rename 0x29 -> 0x28 failed"
        fi
    else
        echo "0x28 did not ACK. Software reset cannot reach a silent chip. Reseat VIN."
    fi
    echo "XSHUT high (right sensor boots again at 0x29)"
    gpioset --mode=exit "$chip" "${line}=1" || true
    sleep 0.1
    echo "bus after left software reset:"
    i2cdetect -y -r 1 || true
    left_ok=0
    right_ok=0
    probe 0x28 && left_ok=1
    probe 0x29 && right_ok=1
fi

echo "== sensor reset done. Jupyter was left running. =="
echo "UU in the grid means a driver already owns that address."
echo "Do not run the data notebook, the deploy loop, and test_two_sensors.py at the same time."
echo "They share this I2C bus. A second reader causes errno 121 (Read failed)."
if [ "$left_ok" -eq 0 ]; then
    echo "0x28 still failed after the I2C software reset."
    echo "Stop the other notebook or test script, click Reset Sensors again, then reseat the left sensor if it still fails."
fi
if [ "$right_ok" -eq 0 ]; then
    echo "0x29 still failed. Click Reset Sensors once more with no other program reading the sensors."
fi
