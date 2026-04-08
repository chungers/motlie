#!/usr/bin/env bash
# launch-ch.sh — Launch one guest VM for the v1.4 extraction harness.
#
# Usage:
#   ./launch-ch.sh --guest alice
#   ./launch-ch.sh --guest bob
#   ./launch-ch.sh --guest alice --no-net
#   ./launch-ch.sh --guest alice --admin-net=tap --egress-net=vhost-user
#   ./launch-ch.sh --guest alice --overlay-size 2G
#   ./launch-ch.sh --guest alice --cloud-init-dir /tmp/alice-cloud-init
#
# Shared built artifacts:
#   artifacts/base/rootfs.squashfs
#   artifacts/base/Image|vmlinux.bin
#
# Per-guest writable runtime overlays are created on each launch under:
#   ${RUNTIME_ROOT:-/tmp/motlie-vmm-v14-runtime}/<guest>/overlay.ext4

set -euo pipefail

copy_overlay_tree() {
    local source_dir="$1"
    local dest_dir="$2"

    if [ -d "$source_dir" ]; then
        mkdir -p "$dest_dir"
        cp -a "$source_dir"/. "$dest_dir"/
    fi
}

seed_mount_paths_from_mounts_yaml() {
    local mounts_yaml="$1"
    local guest_path

    [ -f "$mounts_yaml" ] || return 0

    while read -r guest_path; do
        [ -n "$guest_path" ] || continue
        mkdir -p "$OVERLAY_SEED/upper$guest_path"
        if [[ "$guest_path" == /home/* ]]; then
            mkdir -p "$OVERLAY_SEED/upper$guest_path/.ssh"
            mkdir -p "$OVERLAY_SEED/upper$guest_path/.config"
            chmod 700 "$OVERLAY_SEED/upper$guest_path/.ssh"
        fi
    done < <(sed -n 's/^    guest_path: //p' "$mounts_yaml")
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64) KERNEL_IMAGE="vmlinux.bin" ;;
    aarch64) KERNEL_IMAGE="Image" ;;
    *) echo "ERROR: unsupported host architecture: $HOST_ARCH"; exit 1 ;;
esac

GUEST_NAME=""
ADMIN_NET="tap"
EGRESS_NET="tap"
OVERLAY_SIZE="${OVERLAY_SIZE:-2G}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/motlie-vmm-v14-runtime}"
CLOUD_INIT_DIR=""
SERIAL_BACKEND="${CH_SERIAL_BACKEND:-tty}"
CONSOLE_BACKEND="${CH_CONSOLE_BACKEND:-off}"
VNET_SOCKET=""
SSH_CA_PUBKEY=""
# Per-guest overrides (all derivable from GUEST_NAME if not supplied)
CID=""
HOST_IP=""
GUEST_IP=""
ADMIN_MAC=""
EGRESS_MAC=""
EGRESS_HOST_IP=""
EGRESS_GUEST_IP=""
EGRESS_DNS_IP=""
SSH_USER=""
GUEST_HOSTNAME=""
LOGIN_HOME=""
MOUNT_CONFIG=""
API_SOCKET=""
VSOCK_SOCKET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest) GUEST_NAME="$2"; shift 2 ;;
        --guest=*) GUEST_NAME="${1#*=}"; shift ;;
        --overlay-size) OVERLAY_SIZE="$2"; shift 2 ;;
        --overlay-size=*) OVERLAY_SIZE="${1#*=}"; shift ;;
        --cloud-init-dir) CLOUD_INIT_DIR="$2"; shift 2 ;;
        --cloud-init-dir=*) CLOUD_INIT_DIR="${1#*=}"; shift ;;
        --admin-net) ADMIN_NET="$2"; shift 2 ;;
        --admin-net=*) ADMIN_NET="${1#*=}"; shift ;;
        --egress-net) EGRESS_NET="$2"; shift 2 ;;
        --egress-net=*) EGRESS_NET="${1#*=}"; shift ;;
        --vnet-socket) VNET_SOCKET="$2"; shift 2 ;;
        --vnet-socket=*) VNET_SOCKET="${1#*=}"; shift ;;
        --ssh-ca-pubkey) SSH_CA_PUBKEY="$2"; shift 2 ;;
        --ssh-ca-pubkey=*) SSH_CA_PUBKEY="${1#*=}"; shift ;;
        --cid) CID="$2"; shift 2 ;;
        --cid=*) CID="${1#*=}"; shift ;;
        --host-ip) HOST_IP="$2"; shift 2 ;;
        --host-ip=*) HOST_IP="${1#*=}"; shift ;;
        --guest-ip) GUEST_IP="$2"; shift 2 ;;
        --guest-ip=*) GUEST_IP="${1#*=}"; shift ;;
        --admin-mac) ADMIN_MAC="$2"; shift 2 ;;
        --admin-mac=*) ADMIN_MAC="${1#*=}"; shift ;;
        --egress-mac) EGRESS_MAC="$2"; shift 2 ;;
        --egress-mac=*) EGRESS_MAC="${1#*=}"; shift ;;
        --egress-host-ip) EGRESS_HOST_IP="$2"; shift 2 ;;
        --egress-host-ip=*) EGRESS_HOST_IP="${1#*=}"; shift ;;
        --egress-guest-ip) EGRESS_GUEST_IP="$2"; shift 2 ;;
        --egress-guest-ip=*) EGRESS_GUEST_IP="${1#*=}"; shift ;;
        --egress-dns-ip) EGRESS_DNS_IP="$2"; shift 2 ;;
        --egress-dns-ip=*) EGRESS_DNS_IP="${1#*=}"; shift ;;
        --ssh-user) SSH_USER="$2"; shift 2 ;;
        --ssh-user=*) SSH_USER="${1#*=}"; shift ;;
        --hostname) GUEST_HOSTNAME="$2"; shift 2 ;;
        --hostname=*) GUEST_HOSTNAME="${1#*=}"; shift ;;
        --login-home) LOGIN_HOME="$2"; shift 2 ;;
        --login-home=*) LOGIN_HOME="${1#*=}"; shift ;;
        --mount-config) MOUNT_CONFIG="$2"; shift 2 ;;
        --mount-config=*) MOUNT_CONFIG="${1#*=}"; shift ;;
        --no-net) ADMIN_NET="none"; EGRESS_NET="none"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$GUEST_NAME" ]; then
    echo "ERROR: --guest <name> is required"
    exit 1
fi

case "$ADMIN_NET:$EGRESS_NET" in
    none:none|none:vhost-user|tap:tap|tap:vhost-user) ;;
    *)
        echo "ERROR: supported modes are --admin-net=none --egress-net=none, --admin-net=none --egress-net=vhost-user, --admin-net=tap --egress-net=tap, and --admin-net=tap --egress-net=vhost-user"
        exit 1
        ;;
esac

# Derive defaults from GUEST_NAME when not explicitly supplied.
BASE_ARTIFACTS="$SCRIPT_DIR/artifacts/base"
[ -n "$API_SOCKET" ]      || API_SOCKET="/tmp/motlie-vmm-v14-${GUEST_NAME}-api.sock"
[ -n "$VSOCK_SOCKET" ]    || VSOCK_SOCKET="/tmp/motlie-vmm-v14-${GUEST_NAME}.vsock"
[ -n "$CID" ]             || CID=3
[ -n "$HOST_IP" ]         || HOST_IP="192.168.249.1"
[ -n "$GUEST_IP" ]        || GUEST_IP="192.168.249.2"
[ -n "$ADMIN_MAC" ]       || ADMIN_MAC="12:34:56:78:90:aa"
[ -n "$EGRESS_MAC" ]      || EGRESS_MAC="12:34:56:78:90:ab"
[ -n "$EGRESS_HOST_IP" ]  || EGRESS_HOST_IP="10.0.2.2"
[ -n "$EGRESS_GUEST_IP" ] || EGRESS_GUEST_IP="10.0.2.15"
[ -n "$EGRESS_DNS_IP" ]   || EGRESS_DNS_IP="10.0.2.3"
[ -n "$SSH_USER" ]        || SSH_USER="$GUEST_NAME"
[ -n "$GUEST_HOSTNAME" ]  || GUEST_HOSTNAME="motlie-${GUEST_NAME}"
[ -n "$LOGIN_HOME" ]      || LOGIN_HOME="/home/${GUEST_NAME}"
# MOUNT_CONFIG is only required when --cloud-init-dir is not set (manual mode).
# When repl_host generates the launch script, mounts.yaml is in the cloud-init dir.
[ -n "$VNET_SOCKET" ]     || VNET_SOCKET="/tmp/motlie-vmm-v14-${GUEST_NAME}.sock"
GUEST_OVERLAY_CONTENT="$SCRIPT_DIR/overlay.d/${GUEST_NAME}"

if [ ! -f "$BASE_ARTIFACTS/$KERNEL_IMAGE" ]; then
    echo "ERROR: $BASE_ARTIFACTS/$KERNEL_IMAGE not found."
    echo "Run ./build-guest.sh first."
    exit 1
fi
if [ ! -f "$BASE_ARTIFACTS/rootfs.squashfs" ]; then
    echo "ERROR: $BASE_ARTIFACTS/rootfs.squashfs not found."
    echo "Run ./build-guest.sh first."
    exit 1
fi
if [ -z "$CLOUD_INIT_DIR" ] && [ -n "$MOUNT_CONFIG" ] && [ ! -f "$MOUNT_CONFIG" ]; then
    echo "ERROR: mount config not found at $MOUNT_CONFIG"
    exit 1
fi
if [ -z "$CLOUD_INIT_DIR" ] && [ -z "$MOUNT_CONFIG" ]; then
    echo "ERROR: either --cloud-init-dir or --mount-config is required"
    exit 1
fi
if ! command -v mkfs.ext4 >/dev/null 2>&1; then
    echo "ERROR: mkfs.ext4 not found. Install: sudo apt install e2fsprogs"
    exit 1
fi
if [ -n "$CLOUD_INIT_DIR" ]; then
    if [ ! -d "$CLOUD_INIT_DIR" ]; then
        echo "ERROR: cloud-init dir not found at $CLOUD_INIT_DIR"
        exit 1
    fi
    if [ ! -f "$CLOUD_INIT_DIR/user-data" ]; then
        echo "ERROR: cloud-init dir is missing user-data at $CLOUD_INIT_DIR/user-data"
        exit 1
    fi
    if [ ! -f "$CLOUD_INIT_DIR/meta-data" ]; then
        echo "ERROR: cloud-init dir is missing meta-data at $CLOUD_INIT_DIR/meta-data"
        exit 1
    fi
fi

COMMON_OVERLAY_CONTENT="$SCRIPT_DIR/overlay.d/common"
RUNTIME_DIR="$RUNTIME_ROOT/$GUEST_NAME"
RUNTIME_OVERLAY="$RUNTIME_DIR/overlay.ext4"
OVERLAY_SEED="$RUNTIME_DIR/seed"

# Set umask 022 for the entire overlay seed creation.
# sshd checks ownership+mode on every directory in the path up to /.
# Intermediate dirs created by mkdir -p must be 755 (not 775).
umask 022

rm -rf "$OVERLAY_SEED"
mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vfs"
mkdir -p "$OVERLAY_SEED/work"
mkdir -p "$OVERLAY_SEED/upper/etc"

seed_guest_hosts_file() {
    cat > "$OVERLAY_SEED/upper/etc/hosts" <<EOF
127.0.0.1 localhost
127.0.1.1 $GUEST_HOSTNAME

# IPv6 defaults
::1 localhost ip6-localhost ip6-loopback
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
EOF
}

if [ -n "$CLOUD_INIT_DIR" ]; then
    mkdir -p "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud"
    cp "$CLOUD_INIT_DIR/user-data" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/user-data"
    cp "$CLOUD_INIT_DIR/meta-data" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/meta-data"
    if [ -f "$CLOUD_INIT_DIR/network-config" ]; then
        cp "$CLOUD_INIT_DIR/network-config" "$OVERLAY_SEED/upper/var/lib/cloud/seed/nocloud/network-config"
    fi
    if [ -f "$CLOUD_INIT_DIR/mounts.yaml" ]; then
        cp "$CLOUD_INIT_DIR/mounts.yaml" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"
        seed_mount_paths_from_mounts_yaml "$CLOUD_INIT_DIR/mounts.yaml"
    else
        cp "$MOUNT_CONFIG" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"
        seed_mount_paths_from_mounts_yaml "$MOUNT_CONFIG"
    fi
    printf '%s\n' "$GUEST_HOSTNAME" > "$OVERLAY_SEED/upper/etc/hostname"
    seed_guest_hosts_file
else
    cp "$MOUNT_CONFIG" "$OVERLAY_SEED/upper/etc/motlie-vfs/mounts.yaml"
    printf '%s\n' "$GUEST_HOSTNAME" > "$OVERLAY_SEED/upper/etc/hostname"
    seed_guest_hosts_file
    seed_mount_paths_from_mounts_yaml "$MOUNT_CONFIG"
fi

copy_overlay_tree "$COMMON_OVERLAY_CONTENT" "$OVERLAY_SEED/upper"
copy_overlay_tree "$GUEST_OVERLAY_CONTENT" "$OVERLAY_SEED/upper"

if [ "$EGRESS_NET" = "vhost-user" ]; then
    mkdir -p "$OVERLAY_SEED/upper/etc/motlie-vmm"
    printf '%s\n' "$EGRESS_MAC" > "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.mac"
    chmod 644 "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.mac"
    printf '%s\n' "$EGRESS_GUEST_IP" > "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.ipv4"
    printf '%s\n' "$EGRESS_HOST_IP" > "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.gateway"
    printf '%s\n' "$EGRESS_DNS_IP" > "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.dns"
    chmod 644 \
        "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.ipv4" \
        "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.gateway" \
        "$OVERLAY_SEED/upper/etc/motlie-vmm/egress.dns"
fi

# Inject SSH CA pubkey and per-guest principals for cert-based auth (v1.4+).
# sshd_config references these paths via TrustedUserCAKeys and
# AuthorizedPrincipalsFile directives baked into the guest image.
if [ -n "$SSH_CA_PUBKEY" ]; then
    mkdir -p "$OVERLAY_SEED/upper/etc/ssh/ca"
    printf '%s\n' "$SSH_CA_PUBKEY" > "$OVERLAY_SEED/upper/etc/ssh/ca/user_ca.pub"
    chmod 644 "$OVERLAY_SEED/upper/etc/ssh/ca/user_ca.pub"

    # Each user that the CA can authenticate needs a principals file.
    # sshd requires the directory to be root-owned with mode 755 (not 775).
    mkdir -m 755 -p "$OVERLAY_SEED/upper/etc/ssh/auth_principals"
    printf '%s\n' "$SSH_USER" > "$OVERLAY_SEED/upper/etc/ssh/auth_principals/$SSH_USER"
    printf '%s\n' "$SSH_USER" > "$OVERLAY_SEED/upper/etc/ssh/auth_principals/root"
    chmod 644 "$OVERLAY_SEED/upper/etc/ssh/auth_principals/$SSH_USER"
    chmod 644 "$OVERLAY_SEED/upper/etc/ssh/auth_principals/root"
    echo "  SSH CA:    injected user_ca.pub + auth_principals/$SSH_USER"
fi

mkdir -p "$RUNTIME_DIR"
rm -f "$RUNTIME_OVERLAY"
truncate -s "$OVERLAY_SIZE" "$RUNTIME_OVERLAY"
# Use fakeroot so all overlay files get root:root ownership.
# sshd rejects CA/principals files not owned by root.
fakeroot mkfs.ext4 -F -d "$OVERLAY_SEED" "$RUNTIME_OVERLAY" -q
rm -rf "$OVERLAY_SEED"

if [ ! -e /dev/vhost-vsock ]; then
    echo "Loading vhost_vsock kernel module..."
    sudo modprobe vhost_vsock || {
        echo "ERROR: could not load vhost_vsock module."
        exit 1
    }
fi

CMDLINE="console=ttyS0 root=/dev/vda rootfstype=squashfs ro init=/sbin/overlay-init overlay_root=vdb"
if [ -n "$CLOUD_INIT_DIR" ]; then
    CMDLINE="$CMDLINE ds=nocloud;s=file:///var/lib/cloud/seed/nocloud/"
fi
if [ "$ADMIN_NET" = "tap" ]; then
    if [ "$EGRESS_NET" = "vhost-user" ]; then
        CMDLINE="$CMDLINE ip=${GUEST_IP}:::255.255.255.0::eth0:off"
    else
        CMDLINE="$CMDLINE ip=${GUEST_IP}::${HOST_IP}:255.255.255.0::eth0:off"
    fi
fi

MEMORY_ARG="size=512M"
if [ "$EGRESS_NET" = "vhost-user" ]; then
    MEMORY_ARG="size=512M,shared=on"
fi

CH_ARGS=(
    --api-socket "$API_SOCKET"
    --kernel "$BASE_ARTIFACTS/$KERNEL_IMAGE"
    --cmdline "$CMDLINE"
    --cpus boot=2
    --memory "$MEMORY_ARG"
    --vsock "cid=$CID,socket=$VSOCK_SOCKET"
    --serial "$SERIAL_BACKEND"
    --console "$CONSOLE_BACKEND"
)

DISK_ARGS=(
    "path=$BASE_ARTIFACTS/rootfs.squashfs,readonly=on"
    "path=$RUNTIME_OVERLAY"
)

NET_ARGS=()
if [ "$ADMIN_NET:$EGRESS_NET" = "tap:tap" ]; then
    NET_ARGS+=("tap=,mac=$ADMIN_MAC,ip=$HOST_IP,mask=255.255.255.0")
elif [ "$ADMIN_NET:$EGRESS_NET" = "tap:vhost-user" ]; then
    NET_ARGS+=("tap=,mac=$ADMIN_MAC,ip=$HOST_IP,mask=255.255.255.0")
    NET_ARGS+=("vhost_user=true,socket=$VNET_SOCKET,mac=$EGRESS_MAC")
elif [ "$ADMIN_NET:$EGRESS_NET" = "none:vhost-user" ]; then
    NET_ARGS+=("vhost_user=true,socket=$VNET_SOCKET,mac=$EGRESS_MAC")
fi

rm -f "$API_SOCKET" "$VSOCK_SOCKET"

echo "=== Launching Cloud Hypervisor ($GUEST_NAME) ==="
echo "  Kernel:    $BASE_ARTIFACTS/$KERNEL_IMAGE"
echo "  Squashfs:  $BASE_ARTIFACTS/rootfs.squashfs (vda, ro)"
echo "  Overlay:   $RUNTIME_OVERLAY (vdb, rw runtime)"
echo "  Size:      $OVERLAY_SIZE"
if [ -n "$CLOUD_INIT_DIR" ]; then
    echo "  CloudInit: $CLOUD_INIT_DIR (seeded into /var/lib/cloud/seed/nocloud)"
fi
echo "  vsock:     CID $CID, socket $VSOCK_SOCKET"
echo "  Serial:    $SERIAL_BACKEND"
echo "  Console:   $CONSOLE_BACKEND"
if [ "$ADMIN_NET:$EGRESS_NET" = "tap:tap" ]; then
    echo "  Network:   legacy TAP, guest $GUEST_IP, host $HOST_IP"
    echo "  SSH:       ssh $SSH_USER@$GUEST_IP (password: testpass)"
elif [ "$ADMIN_NET:$EGRESS_NET" = "tap:vhost-user" ]; then
    echo "  Admin NIC: TAP, guest $GUEST_IP, host $HOST_IP"
    echo "  Egress:    vhost-user via $VNET_SOCKET (MAC $EGRESS_MAC)"
    echo "  SSH:       ssh $SSH_USER@$GUEST_IP (password: testpass)"
elif [ "$ADMIN_NET:$EGRESS_NET" = "none:vhost-user" ]; then
    echo "  Network:   rootless/userspace only"
    echo "  Egress:    vhost-user via $VNET_SOCKET (MAC $EGRESS_MAC)"
    echo "  SSH:       proxied only via localhost:2222"
else
    echo "  Network:   disabled (--no-net)"
fi
echo "  Lifetime:  recreated on every launch"
echo "  API:       $API_SOCKET"
echo ""
echo "To stop: curl --unix-socket $API_SOCKET -X PUT http://localhost/api/v1/vm.shutdown"
echo ""

if [ "${#NET_ARGS[@]}" -gt 0 ]; then
    exec cloud-hypervisor "${CH_ARGS[@]}" --disk "${DISK_ARGS[@]}" --net "${NET_ARGS[@]}"
else
    exec cloud-hypervisor "${CH_ARGS[@]}" --disk "${DISK_ARGS[@]}"
fi
