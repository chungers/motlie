#!/usr/bin/env bash
# build-guest.sh — Build the generic shared base image set for v1.2.
#
# Produces:
#   artifacts/base/rootfs.squashfs   — shared Debian rootfs for alice+bob
#   artifacts/base/Image|vmlinux.bin — shared CH-compatible kernel
#
# Guest identity is not baked here. Alice/Bob differences are created by
# launch-ch.sh as runtime writable overlays.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    local install_hint="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        die "$cmd not found. Install: $install_hint"
    fi
}

passwd_primary_gid() {
    getent passwd "$USER" | cut -d: -f4
}

passwd_primary_group() {
    getent group "$(passwd_primary_gid)" | cut -d: -f1
}

check_unshare_prereqs() {
    local current_gid expected_gid expected_group

    current_gid="$(id -g)"
    expected_gid="$(passwd_primary_gid)"
    expected_group="$(passwd_primary_group)"

    if [ -z "$expected_gid" ] || [ -z "$expected_group" ]; then
        die "failed to resolve passwd primary gid/group for $USER"
    fi

    if [ "$current_gid" != "$expected_gid" ]; then
        cat >&2 <<EOF
ERROR: mmdebstrap --mode=unshare requires this shell's primary gid to match the
passwd entry for $USER.

Current shell gid: $current_gid ($(id -gn))
Passwd primary gid: $expected_gid ($expected_group)

Try one of:
  1. Open a fresh login shell for $USER
  2. Run: exec newgrp
  3. Use a rootful fallback: MMDEBSTRAP_MODE=root ./build-guest.sh
EOF
        exit 1
    fi

    if ! grep -q "^${USER}:" /etc/subuid; then
        die "/etc/subuid has no entry for $USER"
    fi
    if ! grep -q "^${USER}:" /etc/subgid; then
        die "/etc/subgid has no entry for $USER"
    fi
}

run_mmdebstrap() {
    local target="$1"
    shift

    local -a cmd=(
        mmdebstrap
        --mode="$MMDEBSTRAP_MODE"
        --arch="$DEBOOTSTRAP_ARCH"
        --variant=minbase
        --format=squashfs
        --keyring=/usr/share/keyrings/debian-archive-keyring.gpg
        "$@"
        "$DEBIAN_SUITE" "$target" "$DEBIAN_MIRROR"
    )

    if [ "$MMDEBSTRAP_MODE" = "unshare" ]; then
        check_unshare_prereqs
        "${cmd[@]}"
        return
    fi

    if [ "$MMDEBSTRAP_MODE" = "root" ] || [ "$MMDEBSTRAP_MODE" = "sudo" ]; then
        if [ "$EUID" -eq 0 ]; then
            "${cmd[@]}"
        else
            sudo "${cmd[@]}"
            sudo chown "$(id -u):$(id -g)" "$target"
        fi
        return
    fi

    "${cmd[@]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

GUEST_BINARY=""
KERNEL_MODE="download"
MMDEBSTRAP_MODE="${MMDEBSTRAP_MODE:-unshare}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest)
            die "v1.2 builds one generic base image; guest selection moved to launch-ch.sh"
            ;;
        --guest-binary) GUEST_BINARY="$2"; shift 2 ;;
        --kernel) KERNEL_MODE="$2"; shift 2 ;;
        --base-only) shift ;;
        --overlay-only)
            die "--overlay-only no longer applies; launch-ch.sh creates per-guest runtime overlays"
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$KERNEL_MODE" in
    download|build|skip) ;;
    *) die "--kernel must be one of: download, build, skip" ;;
esac

case "$MMDEBSTRAP_MODE" in
    auto|sudo|root|unshare|fakeroot|fakechroot|chrootless) ;;
    *) die "MMDEBSTRAP_MODE must be one of: auto, sudo, root, unshare, fakeroot, fakechroot, chrootless" ;;
esac

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)
        RUST_TARGET="x86_64-unknown-linux-gnu"
        DEBOOTSTRAP_ARCH="amd64"
        KERNEL_IMAGE="vmlinux.bin"
        KERNEL_RELEASE_ASSET="vmlinux"
        KERNEL_BUILD_TARGET="bzImage"
        KERNEL_BUILD_OUTPUT="arch/x86/boot/compressed/vmlinux.bin"
        ;;
    aarch64)
        RUST_TARGET="aarch64-unknown-linux-gnu"
        DEBOOTSTRAP_ARCH="arm64"
        KERNEL_IMAGE="Image"
        KERNEL_RELEASE_ASSET="Image-arm64"
        KERNEL_BUILD_TARGET="Image"
        KERNEL_BUILD_OUTPUT="arch/arm64/boot/Image"
        ;;
    *)
        die "unsupported host architecture: $HOST_ARCH"
        ;;
esac

CH_KERNEL_RELEASE="ch-release-v6.16.9-20251112"
CH_KERNEL_URL="https://github.com/cloud-hypervisor/linux/releases/download/${CH_KERNEL_RELEASE}/${KERNEL_RELEASE_ASSET}"

require_cmd mmdebstrap "sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring"
require_cmd tar2sqfs "sudo apt install squashfs-tools-ng"

if [ ! -f /usr/share/keyrings/debian-archive-keyring.gpg ]; then
    die "Debian archive keyring not found. Install: sudo apt install debian-archive-keyring"
fi

APPARMOR_USERNS=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)
if [ "$APPARMOR_USERNS" = "1" ] && [ "$MMDEBSTRAP_MODE" = "unshare" ]; then
    die "AppArmor restricts unprivileged user namespaces. Fix: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
fi

DEBIAN_SUITE="bookworm"
DEBIAN_MIRROR="http://deb.debian.org/debian"
BASE_ARTIFACTS="$SCRIPT_DIR/artifacts/base"
BASE_ROOTFS="$BASE_ARTIFACTS/rootfs.squashfs"
BASE_KERNEL="$BASE_ARTIFACTS/$KERNEL_IMAGE"
BASE_HOSTNAME="motlie-vfs-v12"
EGRESS_MAC="12:34:56:78:90:ab"

echo "=== motlie-vfs v1.2 base image builder ==="
echo "Host arch:          $HOST_ARCH"
echo "Rust target:        $RUST_TARGET"
echo "Debian arch:        $DEBOOTSTRAP_ARCH"
echo "Kernel mode:        $KERNEL_MODE"
echo "mmdebstrap mode:    $MMDEBSTRAP_MODE"
echo "Base dir:           $BASE_ARTIFACTS"
echo ""

mkdir -p "$BASE_ARTIFACTS"

if [ -z "$GUEST_BINARY" ]; then
    echo "--- Step 1: Building motlie-vfs-guest-v1_1 ($RUST_TARGET) ---"
    (cd "$WORKSPACE_ROOT" && cargo build --release \
        --features vsock,client \
        -p motlie-vfs --bin motlie-vfs-guest-v1_1)
    GUEST_BINARY="$WORKSPACE_ROOT/target/release/motlie-vfs-guest-v1_1"
    echo "Guest binary: $GUEST_BINARY"
    file "$GUEST_BINARY"
    echo ""
fi

if [ ! -f "$GUEST_BINARY" ]; then
    die "guest binary not found at $GUEST_BINARY"
fi

echo "--- Step 2: Kernel ($KERNEL_MODE) ---"
case "$KERNEL_MODE" in
    download)
        if [ -f "$BASE_KERNEL" ]; then
            echo "Kernel already exists: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        else
            echo "Downloading pre-built kernel from cloud-hypervisor/linux..."
            wget -q --show-progress -O "$BASE_KERNEL" "$CH_KERNEL_URL"
            echo "Kernel: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        fi
        ;;
    build)
        KERNEL_SRC="/tmp/motlie-vfs-kernel-$$"
        echo "Cloning cloud-hypervisor/linux into $KERNEL_SRC..."
        git clone --depth 1 https://github.com/cloud-hypervisor/linux.git \
            -b ch-6.12.8 "$KERNEL_SRC"
        (
            cd "$KERNEL_SRC"
            make ch_defconfig
            cat >> .config << 'KEOF'
CONFIG_SQUASHFS=y
CONFIG_SQUASHFS_ZSTD=y
CONFIG_OVERLAY_FS=y
KEOF
            make olddefconfig
            make "$KERNEL_BUILD_TARGET" -j"$(nproc)"
            cp "$KERNEL_BUILD_OUTPUT" "$BASE_KERNEL"
        )
        rm -rf "$KERNEL_SRC"
        echo "Kernel: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        ;;
    skip)
        if [ ! -f "$BASE_KERNEL" ]; then
            echo "WARNING: kernel not found at $BASE_KERNEL"
            echo "Place a CH-compatible kernel there before running launch-ch.sh."
        else
            echo "Using existing kernel: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        fi
        ;;
esac
echo ""

echo "--- Step 3: Building shared Debian squashfs root image ---"
GUEST_BINARY_ABS="$(realpath "$GUEST_BINARY")"
OVERLAY_INIT_ABS="$(realpath "$SCRIPT_DIR/overlay-init")"

run_mmdebstrap "$BASE_ROOTFS" \
    --include=openssh-server,bash,ca-certificates,coreutils,curl,dnsutils,tmux,fuse3,libfuse3-3,systemd,systemd-sysv,dbus,iproute2,cloud-init,locales \
    --customize-hook='chroot "$1" systemctl enable ssh' \
    --customize-hook='chroot "$1" systemctl enable systemd-networkd' \
    --customize-hook='chroot "$1" systemctl disable systemd-networkd-wait-online.service' \
    --customize-hook='rm -f "$1/etc/systemd/system/systemd-networkd-wait-online.service" "$1/etc/systemd/system/network-online.target.wants/systemd-networkd-wait-online.service"' \
    --customize-hook='printf "en_US.UTF-8 UTF-8\n" > "$1/etc/locale.gen"' \
    --customize-hook='chroot "$1" locale-gen en_US.UTF-8' \
    --customize-hook='chroot "$1" update-locale LANG=en_US.UTF-8' \
    --customize-hook='chroot "$1" groupadd -g 1000 alice' \
    --customize-hook='chroot "$1" useradd -m -u 1000 -g alice -s /bin/bash alice' \
    --customize-hook='echo "alice:testpass" | chroot "$1" chpasswd' \
    --customize-hook='chroot "$1" groupadd -g 1001 bob' \
    --customize-hook='chroot "$1" useradd -m -u 1001 -g bob -s /bin/bash bob' \
    --customize-hook='echo "bob:testpass" | chroot "$1" chpasswd' \
    --customize-hook='sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" "$1/etc/ssh/sshd_config"' \
    --customize-hook='echo "root:rootpass" | chroot "$1" chpasswd' \
    --customize-hook='chroot "$1" ssh-keygen -A' \
    --customize-hook='mkdir -p "$1/etc/motlie-vfs"' \
    --customize-hook='cat > "$1/etc/profile.d/dotenv.sh" << "DOTENVEOF"
if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
DOTENVEOF' \
    --customize-hook='cat > "$1/etc/profile.d/tmux-auto.sh" << "TMUXEOF"
if [ -n "$SSH_CONNECTION" ] && [ -z "$TMUX" ] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$USER" 2>/dev/null; then
        echo "Attaching to existing tmux session..."
        sleep 1
        exec tmux attach-session -t "$USER"
    else
        printf "Start tmux session? [Y/n] (auto-yes in 3s) "
        if read -r -n 1 -t 3 answer; then
            echo
        else
            answer=Y
            echo
        fi
        case "$answer" in
            n|N) ;;
            *) exec tmux new-session -s "$USER" ;;
        esac
    fi
fi
TMUXEOF' \
    --customize-hook='cat > "$1/etc/profile.d/agent-state.sh" << "AGENTEOF"
agent_state_root=/agent-state
codex_root="$agent_state_root/codex"
codex_sqlite_root="$codex_root/sqlite"
claude_root="$agent_state_root/claude"
claude_code_root="$agent_state_root/claude-code"
if [ -d "$agent_state_root" ] && [ -n "${HOME:-}" ] && [ -d "$HOME" ]; then
    mkdir -p "$codex_root" "$codex_sqlite_root" "$claude_root" "$claude_code_root" "$HOME/.config" >/dev/null 2>&1 || true
    export CODEX_HOME="$codex_root"
    export CODEX_SQLITE_HOME="$codex_sqlite_root"
fi
AGENTEOF' \
    --customize-hook='cat > "$1/usr/local/bin/motlie-agent-state-setup" << "AGENTSVCEOF"
#!/bin/sh
set -eu

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"

    [ -d "$home_dir" ] || return 0

    install -d -m 0755 "$home_dir/.config"
    install -d -m 0700 "$home_dir/.codex" "$home_dir/.claude" "$home_dir/.config/claude-code"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    chown "$user_name:$user_name" \
        "$home_dir/.config" \
        "$home_dir/.codex" \
        "$home_dir/.claude" \
        "$home_dir/.config/claude-code" \
        /agent-state/codex \
        /agent-state/codex/sqlite \
        /agent-state/claude \
        /agent-state/claude-code || true

    mountpoint -q "$home_dir/.codex" || mount --bind /agent-state/codex "$home_dir/.codex"
    mountpoint -q "$home_dir/.claude" || mount --bind /agent-state/claude "$home_dir/.claude"
    mountpoint -q "$home_dir/.config/claude-code" || mount --bind /agent-state/claude-code "$home_dir/.config/claude-code"
}

for user_name in alice bob; do
    if id -u "$user_name" >/dev/null 2>&1; then
        setup_user "$user_name"
    fi
done
AGENTSVCEOF' \
    --customize-hook='chmod 755 "$1/usr/local/bin/motlie-agent-state-setup"' \
    --customize-hook="cat > \"\$1/etc/systemd/network/20-egress.network\" << \"NETEOF\"
[Match]
Name=eth1
MACAddress=$EGRESS_MAC

[Network]
ConfigureWithoutCarrier=yes
NETEOF" \
    --customize-hook='cat > "$1/usr/local/bin/motlie-vnet-egress-setup" << "EGRESSEOF"
#!/bin/sh
set -eu

ip link set eth1 up || exit 0
ip addr replace 10.0.2.15/24 dev eth1
ip route replace 10.0.2.0/24 dev eth1 scope link src 10.0.2.15
ip route replace default via 10.0.2.2 dev eth1 metric 100
ip route del default dev eth0 2>/dev/null || true
cat > /etc/resolv.conf << "RESOLVEOF"
nameserver 10.0.2.3
options edns0
RESOLVEOF
EGRESSEOF' \
    --customize-hook='chmod 755 "$1/usr/local/bin/motlie-vnet-egress-setup"' \
    --customize-hook='cat > "$1/etc/systemd/system/motlie-vfs-guest.service" << "SVCEOF"
[Unit]
Description=motlie-vfs guest filesystem mounter
ConditionPathExists=/etc/motlie-vfs/mounts.yaml
After=local-fs.target

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vfs-guest /etc/motlie-vfs/mounts.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
SVCEOF' \
    --customize-hook='chroot "$1" systemctl enable motlie-vfs-guest' \
    --customize-hook='cat > "$1/etc/systemd/system/motlie-agent-state.service" << "AGENTUNITEOF"
[Unit]
Description=Bind agent state into mounted guest home
After=motlie-vfs-guest.service
Requires=motlie-vfs-guest.service
ConditionPathIsDirectory=/agent-state

[Service]
Type=oneshot
ExecStart=/usr/local/bin/motlie-agent-state-setup
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
AGENTUNITEOF' \
    --customize-hook='chroot "$1" systemctl enable motlie-agent-state' \
    --customize-hook='cat > "$1/etc/systemd/system/motlie-vnet-egress.service" << "EGRESSUNITEOF"
[Unit]
Description=Configure static v1.2 egress NIC
After=systemd-networkd.service
Wants=systemd-networkd.service
ConditionPathExists=/sys/class/net/eth1

[Service]
Type=oneshot
ExecStart=/usr/local/bin/motlie-vnet-egress-setup
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EGRESSUNITEOF' \
    --customize-hook='chroot "$1" systemctl enable motlie-vnet-egress' \
    --customize-hook="echo \"$BASE_HOSTNAME\" > \"\$1/etc/hostname\"" \
    --customize-hook='cat > "$1/etc/motd" << "MOTDEOF"
                    _   _ _
  _ __ ___   ___ | |_| (_) ___
 | '"'"'_ ` _ \ / _ \| __| | |/ _ \
 | | | | | | (_) | |_| | |  __/
 |_| |_| |_|\___/ \__|_|_|\___|

v1.2 split-network / agent-state demo
MOTDEOF' \
    --customize-hook='cat > "$1/etc/fstab" << "FSTABEOF"
proc        /proc    proc    defaults        0      0
sysfs       /sys     sysfs   defaults        0      0
devtmpfs    /dev     devtmpfs defaults       0      0
FSTABEOF' \
    --customize-hook='echo "user_allow_other" >> "$1/etc/fuse.conf"' \
    --customize-hook="upload $GUEST_BINARY_ABS /usr/local/bin/motlie-vfs-guest" \
    --customize-hook="chmod 755 \"\$1/usr/local/bin/motlie-vfs-guest\"" \
    --customize-hook="upload $OVERLAY_INIT_ABS /sbin/overlay-init" \
    --customize-hook="chmod 755 \"\$1/sbin/overlay-init\"" \
    --customize-hook='chroot "$1" apt-get clean' \
    --customize-hook='rm -rf "$1/var/lib/apt/lists"/*'

echo "Shared squashfs image: $BASE_ROOTFS ($(du -h "$BASE_ROOTFS" | cut -f1))"
echo ""
echo "=== Build complete ==="
ls -lh "$BASE_ARTIFACTS/"
echo ""
echo "Next: ./launch-ch.sh --guest alice --admin-net=tap --egress-net=tap"
