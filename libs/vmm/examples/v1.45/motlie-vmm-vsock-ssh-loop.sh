#!/bin/sh
set -eu

while true; do
    /usr/bin/socat VSOCK-CONNECT:2:2222 TCP:127.0.0.1:22 || true
    sleep 1
done
