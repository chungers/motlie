#!/usr/bin/env bash
set -euo pipefail

start="$(date +%s)"
printf 'testpass\n' | sudo -S apt-get update -o Acquire::Retries=0 >/tmp/apt-update.log 2>&1
end="$(date +%s)"

echo "elapsed:$((end-start))"
tail -n 40 /tmp/apt-update.log
