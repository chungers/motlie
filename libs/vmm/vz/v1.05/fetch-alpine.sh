#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${ALPINE_BASE_URL:-https://dl-cdn.alpinelinux.org/alpine/v3.22/releases/aarch64/netboot}"

curl -fL "$BASE_URL/vmlinuz-virt" -o "$ROOT/vmlinuz-virt"
curl -fL "$BASE_URL/initramfs-virt" -o "$ROOT/initramfs-virt"
