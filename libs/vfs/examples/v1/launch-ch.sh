#!/usr/bin/env bash
# launch-ch.sh — Launch a Cloud Hypervisor guest for motlie-vfs testing.
#
# Prerequisites:
#   - Cloud Hypervisor installed (https://github.com/cloud-hypervisor/cloud-hypervisor)
#   - artifacts/vmlinux.bin or Image (CH-compatible kernel)
#   - artifacts/rootfs.squashfs (built by build-guest.sh)
#   - artifacts/overlay.ext4 (built by build-guest.sh)
#   - /dev/vhost-vsock accessible (sudo modprobe vhost_vsock)
#   - For --pasta: passt package installed (apt install passt)
#   - For default TAP: CAP_NET_ADMIN (sudo setcap or run with sudo)
#
# Usage:
#   ./launch-ch.sh              # TAP networking (needs CAP_NET_ADMIN)
#   ./launch-ch.sh --pasta      # pasta networking (rootless, no capabilities needed)
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
NET_MODE="tap"   # tap (requires CAP_NET_ADMIN) or pasta (rootless)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-net) USE_NET=false; shift ;;
        --pasta) NET_MODE="pasta"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate artifacts exist
# ---------------------------------------------------------------------------
for f in "$KERNEL_IMAGE" rootfs.squashfs overlay.ext4; do
    if [ ! -f "$ARTIFACTS/$f" ]; then
        echo "ERROR: $ARTIFACTS/$f not found."
        echo "Run ./build-guest.sh first, and place vmlinux.bin in artifacts/."
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
# root=/dev/vda  — the squashfs is the first --disk entry
# rootfstype=squashfs ro — mount as read-only squashfs
# init=/sbin/overlay-init — our custom init that stacks overlayfs
# overlay_root=vdb — tells overlay-init which device is the ext4 overlay
# console=hvc0 — serial console for boot messages
CMDLINE="console=hvc0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb"

if $USE_NET && [ "$NET_MODE" = "tap" ]; then
    # Static IP for the guest via kernel ip= parameter
    # Format: ip=<client-ip>::<gateway>:<netmask>::<device>:off
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
    # Disk 1: squashfs root (vda, read-only)
    # Disk 2: ext4 overlay (vdb, read-write)
    --disk "path=$ARTIFACTS/rootfs.squashfs,readonly=on"
           "path=$ARTIFACTS/overlay.ext4"
    # vsock: CID 3, socket at /tmp/motlie-vfs.vsock
    # Host listens on /tmp/motlie-vfs.vsock_<port> for guest-initiated connections
    # Host connects via /tmp/motlie-vfs.vsock with "CONNECT <port>\n" prefix
    --vsock "cid=3,socket=/tmp/motlie-vfs.vsock"
    # Serial console on the terminal
    --serial tty
    --console off
)

if $USE_NET && [ "$NET_MODE" = "tap" ]; then
    # TAP networking: CH auto-creates a vmtapX device
    # Guest is 192.168.249.2, host is 192.168.249.1
    # Requires CAP_NET_ADMIN or sudo
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
    echo "  Network:   $NET_MODE"
    if [ "$NET_MODE" = "tap" ]; then
        echo "  SSH:       ssh alice@192.168.249.2 (password: testpass)"
    else
        echo "  SSH:       guest gets DHCP via pasta (check guest IP with 'ip addr')"
    fi
fi
echo "  API:       /tmp/motlie-vfs-api.sock"
echo ""
echo "To stop: curl --unix-socket /tmp/motlie-vfs-api.sock -X PUT http://localhost/api/v1/vm.shutdown"
echo ""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if $USE_NET && [ "$NET_MODE" = "pasta" ]; then
    # pasta creates a network namespace with a TAP device and translates
    # traffic to host sockets — no CAP_NET_ADMIN or root needed on the host.
    #
    # How it works:
    # 1. pasta creates a user+network namespace with a configured TAP
    # 2. Inside the namespace, we open the TAP fd and pass it to CH
    # 3. CH uses the pre-opened TAP fd (no CAP_NET_ADMIN needed)
    # 4. pasta translates L2 frames to host L4 sockets
    #
    # Prerequisites:
    #   - passt package installed (apt install passt)
    #   - Unprivileged user namespaces allowed:
    #     sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
    #
    # The guest uses DHCP to get its IP from pasta's built-in DHCP server.

    if ! command -v pasta &>/dev/null; then
        echo "ERROR: pasta not found. Install with: apt install passt"
        exit 1
    fi

    # Check if user namespaces are allowed
    APPARMOR_USERNS=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)
    if [ "$APPARMOR_USERNS" = "1" ]; then
        echo "ERROR: AppArmor restricts unprivileged user namespaces."
        echo "Fix:   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
        exit 1
    fi

    # EXPERIMENTAL: pasta creates a network namespace and CH creates its
    # own TAP inside it. Guest uses DHCP.
    #
    # Known limitation: CH's TAP and pasta's TAP are separate interfaces
    # in the namespace. Traffic routing between them may require bridging
    # or further configuration. See PLAN-v1.md for the v1.5 embedded
    # vhost-user-net approach which avoids this issue entirely.

    if ! command -v pasta &>/dev/null; then
        echo "ERROR: pasta not found. Install with: apt install passt"
        exit 1
    fi

    APPARMOR_USERNS=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)
    if [ "$APPARMOR_USERNS" = "1" ]; then
        echo "ERROR: AppArmor restricts unprivileged user namespaces."
        echo "Fix:   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
        exit 1
    fi

    # CH inside the namespace creates its own TAP named "vmtap0"
    pasta --config-net -- \
        cloud-hypervisor \
            --api-socket /tmp/motlie-vfs-api.sock \
            --kernel "$ARTIFACTS/$KERNEL_IMAGE" \
            --cmdline "console=hvc0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb ip=dhcp" \
            --cpus boot=2 \
            --memory size=512M \
            --disk "path=$ARTIFACTS/rootfs.squashfs,readonly=on" \
                   "path=$ARTIFACTS/overlay.ext4" \
            --vsock "cid=3,socket=/tmp/motlie-vfs.vsock" \
            --net "tap=vmtap0,mac=12:34:56:78:90:ab" \
            --serial tty \
            --console off
else
    cloud-hypervisor "${CH_ARGS[@]}"
fi
