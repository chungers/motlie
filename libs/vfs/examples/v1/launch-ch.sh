#!/usr/bin/env bash
# launch-ch.sh — Launch a Cloud Hypervisor guest for motlie-vfs testing.
#
# Prerequisites:
#   - Cloud Hypervisor installed (https://github.com/cloud-hypervisor/cloud-hypervisor)
#   - artifacts/vmlinux.bin or Image (CH-compatible kernel)
#   - artifacts/rootfs.squashfs (built by build-guest.sh)
#   - artifacts/overlay.ext4 (built by build-guest.sh)
#   - /dev/vhost-vsock accessible (sudo modprobe vhost_vsock)
#   - CAP_NET_ADMIN for TAP networking (sudo setcap cap_net_admin+ep $(which cloud-hypervisor))
#
# Usage:
#   ./launch-ch.sh              # TAP networking (needs CAP_NET_ADMIN)
#   ./launch-ch.sh --no-net     # vsock only, no networking
#
# Guest will boot with:
#   /dev/vda = rootfs.squashfs (read-only)
#   /dev/vdb = overlay.ext4 (read-write)
#   vsock CID 3 on /tmp/motlie-vfs.vsock
#   TAP networking at 192.168.249.2 (unless --no-net)
#
# To stop the guest:
#   curl --unix-socket /tmp/motlie-vfs-api.sock -X PUT http://localhost/api/v1/vm.shutdown
#   # or from inside the guest: poweroff

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS="$SCRIPT_DIR/artifacts"

# ---------------------------------------------------------------------------
# Detect host architecture
# ---------------------------------------------------------------------------
HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)
        KERNEL_IMAGE="vmlinux.bin"
        ;;
    aarch64)
        # CH on aarch64 expects an uncompressed Image or PE binary
        KERNEL_IMAGE="Image"
        ;;
    *)
        echo "ERROR: unsupported host architecture: $HOST_ARCH"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_NET=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-net) USE_NET=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate artifacts exist
# ---------------------------------------------------------------------------
for f in "$KERNEL_IMAGE" rootfs.squashfs overlay.ext4; do
    if [ ! -f "$ARTIFACTS/$f" ]; then
        echo "ERROR: $ARTIFACTS/$f not found."
        echo "Run ./build-guest.sh first, and place the kernel in artifacts/."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Ensure vhost-vsock module is loaded
# ---------------------------------------------------------------------------
if [ ! -e /dev/vhost-vsock ]; then
    echo "Loading vhost_vsock kernel module..."
    sudo modprobe vhost_vsock || {
        echo "ERROR: could not load vhost_vsock module."
        echo "Ensure your host kernel has CONFIG_VHOST_VSOCK=m or =y."
        exit 1
    }
fi

# ---------------------------------------------------------------------------
# Build the kernel command line
# ---------------------------------------------------------------------------
CMDLINE="console=hvc0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb"

if $USE_NET; then
    CMDLINE="$CMDLINE ip=192.168.249.2::192.168.249.1:255.255.255.0::eth0:off"
fi

# ---------------------------------------------------------------------------
# Build the CH command
# ---------------------------------------------------------------------------
CH_ARGS=(
    --api-socket /tmp/motlie-vfs-api.sock
    --kernel "$ARTIFACTS/$KERNEL_IMAGE"
    --cmdline "$CMDLINE"
    --cpus boot=2
    --memory size=512M
    --disk "path=$ARTIFACTS/rootfs.squashfs,readonly=on"
           "path=$ARTIFACTS/overlay.ext4"
    --vsock "cid=3,socket=/tmp/motlie-vfs.vsock"
    --serial tty
    --console off
)

if $USE_NET; then
    CH_ARGS+=(--net "tap=,mac=12:34:56:78:90:ab,ip=192.168.249.1,mask=255.255.255.0")
fi

# ---------------------------------------------------------------------------
# Clean up stale sockets
# ---------------------------------------------------------------------------
rm -f /tmp/motlie-vfs-api.sock /tmp/motlie-vfs.vsock

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "=== Launching Cloud Hypervisor ==="
echo "  Arch:      $HOST_ARCH"
echo "  Kernel:    $ARTIFACTS/$KERNEL_IMAGE"
echo "  Squashfs:  $ARTIFACTS/rootfs.squashfs (vda, ro)"
echo "  Overlay:   $ARTIFACTS/overlay.ext4 (vdb, rw)"
echo "  vsock:     CID 3, socket /tmp/motlie-vfs.vsock"
if $USE_NET; then
    echo "  Network:   TAP, guest 192.168.249.2, host 192.168.249.1"
    echo "  SSH:       ssh alice@192.168.249.2 (password: testpass)"
fi
echo "  API:       /tmp/motlie-vfs-api.sock"
echo ""
echo "To stop: curl --unix-socket /tmp/motlie-vfs-api.sock -X PUT http://localhost/api/v1/vm.shutdown"
echo ""

cloud-hypervisor "${CH_ARGS[@]}"
