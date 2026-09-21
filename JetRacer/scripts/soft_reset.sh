#!/usr/bin/env bash
# Soft reset for JetRacer. Run as root (sudo / systemd-run).
# Jupyter is stopped LAST so display / nvargus / cache still run after the
# kernel that clicked the button is gone.
set -u
LOG=/tmp/jetracer_soft_reset.log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) soft reset start =="

sleep 2

echo "stop OLED display"
systemctl stop jetcard_display.service || true

echo "restart camera daemon"
systemctl restart nvargus-daemon || true

echo "drop disk cache"
sync
echo 3 > /proc/sys/vm/drop_caches || true

echo "stop leftover kernels"
pkill -9 -f ipykernel || true

echo "stop Jupyter (page will disconnect — this is expected)"
systemctl stop jetcard_jupyter.service || true

sleep 2

echo "start OLED display"
systemctl start jetcard_display.service || true

echo "start Jupyter"
systemctl start jetcard_jupyter.service || true

echo "== $(date) soft reset done — refresh the browser =="
