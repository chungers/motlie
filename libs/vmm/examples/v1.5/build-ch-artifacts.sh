#!/usr/bin/env bash
# build-ch-artifacts.sh -- Linux Cloud Hypervisor artifact emitter for v1.5.
#
# CONVERGENCE CONTRACT:
# - builds the VMM-owned v1.5 guest mounter once during image assembly
# - installs the same guest-visible paths and service graph used by VZ
# - emits artifacts/base/rootfs.squashfs, Image|vmlinux.bin, guest-contract.json
# - launch-ch.sh and guest boot must not run cargo, rustup, npm, apt, or repair
#
# This is intentionally a Linux-only emitter. macOS Apple VZ packaging currently
# remains in build-guest.sh until the Rust image builder can drive both emitters.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    local install_hint="${2:-}"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        if [ -n "$install_hint" ]; then
            die "$cmd not found. Install: $install_hint"
        fi
        die "$cmd not found"
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
  3. Use a rootful fallback: MMDEBSTRAP_MODE=root ./build-ch-artifacts.sh
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

write_guest_contract_json() {
    local output_path="$1"

    cat > "$output_path" <<EOF
{
  "contract_version": "$MOTLIE_V15_CONTRACT_VERSION",
  "packaging_backend": "ch-squashfs",
  "guest_arch": "$DEBOOTSTRAP_ARCH",
  "host_arch": "$HOST_ARCH",
  "kernel_image": "$KERNEL_IMAGE",
  "build_git_sha": "$IMAGE_BUILD_GIT_SHA",
  "build_time_utc": "$IMAGE_BUILD_TIME_UTC",
  "guest_contract": {
    "motlie_vfs_guest_path": "$MOTLIE_V15_GUEST_BIN_OPT",
    "motlie_vfs_guest_compat_path": "$MOTLIE_V15_GUEST_BIN_COMPAT",
    "motlie_vfs_guest_marker": "$MOTLIE_V15_GUEST_MOUNTER_MARKER",
    "motlie_vfs_guest_build_features": "$MOTLIE_V15_GUEST_BUILD_FEATURES",
    "backend_env": "$MOTLIE_V15_BACKEND_ENV_PATH",
    "mounts": "$MOTLIE_V15_MOUNTS_PATH",
    "agent_state": "/agent-state"
  },
  "launch_contract": {
    "builds_allowed": false,
    "forbidden_first_contact_tools": ["apt", "apt-get", "cargo", "npm", "rustup"]
  }
}
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/common-contract.sh"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
IMAGE_BUILD_GIT_SHA="$(git -C "$WORKSPACE_ROOT" rev-parse HEAD)"
IMAGE_BUILD_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$(uname -s)" != "Linux" ]; then
    die "build-ch-artifacts.sh must run on Linux. Use build-guest.sh for the current macOS/VZ packaging path."
fi

GUEST_BINARY=""
ALLOW_GUEST_BINARY_OVERRIDE="${MOTLIE_V15_ALLOW_GUEST_BINARY_OVERRIDE:-0}"
KERNEL_MODE="${MOTLIE_V15_CH_KERNEL_MODE:-download}"
ROOTFS_MODE="${MOTLIE_V15_CH_ROOTFS_MODE:-build}"
MMDEBSTRAP_MODE="${MMDEBSTRAP_MODE:-unshare}"
INSTALL_AGENT_CLIS="${MOTLIE_V15_INSTALL_AGENT_CLIS:-1}"
PACKAGE_INCLUDE="${MOTLIE_V15_PACKAGE_INCLUDE:-openssh-server,bash,bubblewrap,ca-certificates,coreutils,curl,dnsutils,git,tmux,vim,fuse3,libfuse3-3,systemd,systemd-sysv,dbus,iproute2,iputils-ping,cloud-init,locales,sudo,python3,npm,strace,socat,wget}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guest)
            die "v1.5 builds one generic base image; guest identity is seed/overlay data"
            ;;
        --guest-binary)
            [ "$ALLOW_GUEST_BINARY_OVERRIDE" = "1" ] || die "--guest-binary is disabled by default; set MOTLIE_V15_ALLOW_GUEST_BINARY_OVERRIDE=1 only for a fresh CI-produced v1.5 binary"
            GUEST_BINARY="$2"
            shift 2
            ;;
        --guest-binary=*)
            [ "$ALLOW_GUEST_BINARY_OVERRIDE" = "1" ] || die "--guest-binary is disabled by default; set MOTLIE_V15_ALLOW_GUEST_BINARY_OVERRIDE=1 only for a fresh CI-produced v1.5 binary"
            GUEST_BINARY="${1#*=}"
            shift
            ;;
        --kernel) KERNEL_MODE="$2"; shift 2 ;;
        --kernel=*) KERNEL_MODE="${1#*=}"; shift ;;
        --rootfs) ROOTFS_MODE="$2"; shift 2 ;;
        --rootfs=*) ROOTFS_MODE="${1#*=}"; shift ;;
        --base-only) shift ;;
        --no-agent-clis) INSTALL_AGENT_CLIS=0; shift ;;
        --overlay-only)
            die "--overlay-only does not apply; launch-ch.sh creates per-guest runtime overlays"
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "$KERNEL_MODE" in
    download|skip) ;;
    *) die "--kernel must be one of: download, skip" ;;
esac

case "$ROOTFS_MODE" in
    build|skip) ;;
    *) die "--rootfs must be one of: build, skip" ;;
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
        ;;
    aarch64|arm64)
        RUST_TARGET="aarch64-unknown-linux-gnu"
        DEBOOTSTRAP_ARCH="arm64"
        KERNEL_IMAGE="Image"
        KERNEL_RELEASE_ASSET="Image-arm64"
        ;;
    *)
        die "unsupported host architecture: $HOST_ARCH"
        ;;
esac

CH_KERNEL_RELEASE="${CH_KERNEL_RELEASE:-ch-release-v6.16.9-20251112}"
CH_KERNEL_URL="${CH_KERNEL_URL:-https://github.com/cloud-hypervisor/linux/releases/download/${CH_KERNEL_RELEASE}/${KERNEL_RELEASE_ASSET}}"
DEBIAN_SUITE="${DEBIAN_SUITE:-bookworm}"
DEBIAN_MIRROR="${DEBIAN_MIRROR:-http://deb.debian.org/debian}"
ARTIFACTS_DIR="${MOTLIE_V15_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}"
BASE_ARTIFACTS="$ARTIFACTS_DIR/base"
BASE_ROOTFS="$BASE_ARTIFACTS/rootfs.squashfs"
BASE_KERNEL="$BASE_ARTIFACTS/$KERNEL_IMAGE"
BASE_CONTRACT_JSON="$BASE_ARTIFACTS/guest-contract.json"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/motlie-v15-ch.XXXXXX")"

cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
BACKEND_ENV_FILE="$SCRIPT_DIR/backend.env.ch.example"
DATASOURCE_CFG_FILE="$SCRIPT_DIR/99_motlie_ch.cfg"
AGENT_STATE_SETUP_FILE="$SCRIPT_DIR/motlie-agent-state-setup.sh"
AGENT_STATE_UNIT_FILE="$SCRIPT_DIR/motlie-agent-state.service"
SSH_BRIDGE_LOOP_FILE="$SCRIPT_DIR/motlie-vmm-vsock-ssh-loop.sh"
SSH_BRIDGE_UNIT_FILE="$SCRIPT_DIR/motlie-vmm-vsock-ssh.service"
EGRESS_SETUP_FILE="$SCRIPT_DIR/motlie-vmm-egress-setup.sh"
EGRESS_UNIT_FILE="$SCRIPT_DIR/motlie-vmm-egress.service"
OVERLAY_INIT_FILE="$SCRIPT_DIR/overlay-init"
RUNTIME_CONTRACT_JSON="$STAGING_DIR/guest-contract.json"
TMUX_PROFILE="$STAGING_DIR/tmux-auto.sh"
DOTENV_PROFILE="$STAGING_DIR/dotenv.sh"
AGENT_PROFILE="$STAGING_DIR/agent-state.sh"
APT_IPV4_CONF="$STAGING_DIR/99motlie-force-ipv4"
MOTD_FILE="$STAGING_DIR/motd"

require_cmd git
require_cmd cargo
require_cmd file
require_cmd realpath
require_cmd python3
require_cmd mmdebstrap "sudo apt install mmdebstrap squashfs-tools-ng e2fsprogs uidmap debian-archive-keyring"
require_cmd tar2sqfs "sudo apt install squashfs-tools-ng"
if [ "$KERNEL_MODE" = "download" ]; then
    require_cmd wget "sudo apt install wget"
fi
if [ -z "$GUEST_BINARY" ]; then
    require_cmd cc "sudo apt install build-essential"
    require_cmd pkg-config "sudo apt install pkg-config libfuse3-dev"
    if ! pkg-config --exists fuse3; then
        die "fuse3 development metadata not found. Install: sudo apt install libfuse3-dev"
    fi
fi

if [ ! -f /usr/share/keyrings/debian-archive-keyring.gpg ]; then
    die "Debian archive keyring not found. Install: sudo apt install debian-archive-keyring"
fi

APPARMOR_USERNS="$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)"
if [ "$APPARMOR_USERNS" = "1" ] && [ "$MMDEBSTRAP_MODE" = "unshare" ]; then
    die "AppArmor restricts unprivileged user namespaces. Fix: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
fi

for required_file in \
    "$SERVICE_FILE" \
    "$BACKEND_ENV_FILE" \
    "$DATASOURCE_CFG_FILE" \
    "$AGENT_STATE_SETUP_FILE" \
    "$AGENT_STATE_UNIT_FILE" \
    "$SSH_BRIDGE_LOOP_FILE" \
    "$SSH_BRIDGE_UNIT_FILE" \
    "$EGRESS_SETUP_FILE" \
    "$EGRESS_UNIT_FILE" \
    "$OVERLAY_INIT_FILE"
do
    [ -f "$required_file" ] || die "missing v1.5 CH build input: $required_file"
done

mkdir -p "$BASE_ARTIFACTS"

echo "=== motlie-vmm v1.5 CH artifact builder ==="
echo "Host arch:          $HOST_ARCH"
echo "Rust target:        $RUST_TARGET"
echo "Debian arch:        $DEBOOTSTRAP_ARCH"
echo "Kernel mode:        $KERNEL_MODE"
echo "Rootfs mode:        $ROOTFS_MODE"
echo "mmdebstrap mode:    $MMDEBSTRAP_MODE"
echo "Agent CLIs:         $INSTALL_AGENT_CLIS"
echo "Base dir:           $BASE_ARTIFACTS"
echo ""

if [ -z "$GUEST_BINARY" ]; then
    echo "--- Step 1: Building VMM-owned motlie-vfs-guest-v1_5 ($RUST_TARGET) ---"
    (
        cd "$WORKSPACE_ROOT"
        cargo build --manifest-path "$WORKSPACE_ROOT/libs/vmm/Cargo.toml" \
            --release \
            --target "$RUST_TARGET" \
            --no-default-features \
            --features guest-vfs \
            --bin motlie-vfs-guest-v1_5
    )
    GUEST_BINARY="$WORKSPACE_ROOT/target/$RUST_TARGET/release/motlie-vfs-guest-v1_5"
fi

[ -x "$GUEST_BINARY" ] || die "guest binary not found or not executable: $GUEST_BINARY"
GUEST_BINARY_ABS="$(realpath "$GUEST_BINARY")"
GUEST_MARKER="$("$GUEST_BINARY_ABS" --contract 2>/dev/null || true)"
if [ "$GUEST_MARKER" != "$MOTLIE_V15_GUEST_MOUNTER_MARKER" ]; then
    die "guest binary contract marker mismatch: expected $MOTLIE_V15_GUEST_MOUNTER_MARKER, got '${GUEST_MARKER:-<empty>}'"
fi
echo "Guest binary:       $GUEST_BINARY_ABS"
file "$GUEST_BINARY_ABS"
echo ""

write_guest_contract_json "$RUNTIME_CONTRACT_JSON"
motlie_v15_require_guest_contract_json "$RUNTIME_CONTRACT_JSON" "staged CH guest contract"

cat > "$TMUX_PROFILE" <<'EOF'
# Auto-start tmux only for real interactive Bash SSH logins.
[ -n "${BASH_VERSION:-}" ] || return 0
case $- in
    *i*) ;;
    *) return 0 ;;
esac
[ -n "${SSH_CONNECTION:-}" ] || return 0
[ -z "${TMUX:-}" ] || return 0
[ -t 0 ] && [ -t 1 ] || return 0
command -v tmux >/dev/null 2>&1 || return 0
case "${TERM:-}" in
    ""|dumb|unknown) return 0 ;;
esac
command -v infocmp >/dev/null 2>&1 || return 0
infocmp "$TERM" >/dev/null 2>&1 || return 0

if tmux has-session -t "$USER" 2>/dev/null; then
    echo "Attaching to existing tmux session..."
    sleep 1
    tmux attach-session -t "$USER" || echo "tmux attach failed; continuing without tmux"
    return 0
fi

printf "Start tmux session? [Y/n] (auto-yes in 3s) "
if IFS= read -r -n 1 -t 3 answer; then
    echo
else
    answer=Y
    echo
fi

case "$answer" in
    n|N) ;;
    *) tmux new-session -s "$USER" || echo "tmux start failed; continuing without tmux" ;;
esac
EOF

cat > "$DOTENV_PROFILE" <<'EOF'
if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
EOF

cat > "$AGENT_PROFILE" <<'EOF'
agent_state_root=/agent-state
codex_root="$agent_state_root/codex"
codex_sqlite_root="$codex_root/sqlite"
claude_root="$agent_state_root/claude"
claude_code_root="$agent_state_root/claude-code"
if [ -d "$agent_state_root" ] && [ -n "${HOME:-}" ] && [ -d "$HOME" ] && [ "${USER:-}" != "root" ] && [ "${HOME#"/home/"}" != "$HOME" ]; then
    mkdir -p "$codex_root" "$codex_sqlite_root" "$claude_root" "$claude_code_root" "$HOME/.config" >/dev/null 2>&1 || true
    export CODEX_HOME="$codex_root"
    export CODEX_SQLITE_HOME="$codex_sqlite_root"
fi
EOF

cat > "$APT_IPV4_CONF" <<'EOF'
Acquire::ForceIPv4 "true";
EOF

cat > "$MOTD_FILE" <<EOF
                    _   _ _
  _ __ ___   ___ | |_| (_) ___
 | '_ \` _ \ / _ \| __| | |/ _ \\
 | | | | | | (_) | |_| | |  __/
 |_| |_| |_|\___/ \__|_|_|\___|

v1.5 common CH/VZ guest image
Build: $IMAGE_BUILD_GIT_SHA
Built At: $IMAGE_BUILD_TIME_UTC
EOF

echo "--- Step 2: Kernel ($KERNEL_MODE) ---"
case "$KERNEL_MODE" in
    download)
        if [ -f "$BASE_KERNEL" ]; then
            echo "Kernel already exists: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        else
            echo "Downloading CH kernel from $CH_KERNEL_URL"
            wget -q --show-progress -O "$BASE_KERNEL" "$CH_KERNEL_URL"
            echo "Kernel: $BASE_KERNEL ($(du -h "$BASE_KERNEL" | cut -f1))"
        fi
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

if [ "$ROOTFS_MODE" = "build" ]; then
    echo "--- Step 3: Building shared Debian squashfs root image ---"
    rm -f "$BASE_ROOTFS"

    ROOTFS_ARGS=(
        --include="$PACKAGE_INCLUDE"
        --customize-hook='chroot "$1" systemctl enable ssh'
        --customize-hook='chroot "$1" systemctl enable systemd-networkd'
        --customize-hook='chroot "$1" systemctl disable systemd-networkd-wait-online.service'
        --customize-hook='rm -f "$1/etc/systemd/system/systemd-networkd-wait-online.service" "$1/etc/systemd/system/network-online.target.wants/systemd-networkd-wait-online.service"'
        --customize-hook='printf "en_US.UTF-8 UTF-8\n" > "$1/etc/locale.gen"'
        --customize-hook='chroot "$1" locale-gen en_US.UTF-8'
        --customize-hook='chroot "$1" update-locale LANG=en_US.UTF-8'
        --customize-hook='mkdir -p "$1/etc/motlie/v1.5" "$1/etc/motlie-vfs" "$1/etc/ssh/ca" "$1/etc/ssh/auth_principals" "$1/opt/motlie/v1.5/guest/bin" "$1/etc/profile.d" "$1/etc/motlie-vmm"'
        --customize-hook='sed -i "s/#PermitRootLogin.*/PermitRootLogin prohibit-password/" "$1/etc/ssh/sshd_config"'
        --customize-hook='chroot "$1" ssh-keygen -A'
        --customize-hook='cat >> "$1/etc/ssh/sshd_config" << "SSHCAEOF"

# motlie-vmm SSH CA trust (v1.5 common image)
# The CA public key and per-user principals are dynamic seed/overlay data.
TrustedUserCAKeys /etc/ssh/ca/user_ca.pub
AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u
SSHCAEOF'
        --customize-hook='cat > "$1/etc/sudoers.d/90-motlie-vmm" << "SUDOERSEOF"
%sudo ALL=(ALL) NOPASSWD:ALL
SUDOERSEOF
chmod 0440 "$1/etc/sudoers.d/90-motlie-vmm"'
        --customize-hook='systemctl --root="$1" disable apt-daily.service apt-daily.timer apt-daily-upgrade.service apt-daily-upgrade.timer unattended-upgrades.service >/dev/null 2>&1 || true'
        --customize-hook='systemctl --root="$1" mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service >/dev/null 2>&1 || true'
        --customize-hook='echo "user_allow_other" >> "$1/etc/fuse.conf"'
        --customize-hook="upload $GUEST_BINARY_ABS $MOTLIE_V15_GUEST_BIN_OPT"
        --customize-hook="upload $GUEST_BINARY_ABS $MOTLIE_V15_GUEST_BIN_COMPAT"
        --customize-hook="chmod 755 \"\$1$MOTLIE_V15_GUEST_BIN_OPT\" \"\$1$MOTLIE_V15_GUEST_BIN_COMPAT\""
        --customize-hook="chroot \"\$1\" $MOTLIE_V15_GUEST_BIN_OPT --contract | grep -qx '$MOTLIE_V15_GUEST_MOUNTER_MARKER'"
        --customize-hook="upload $SERVICE_FILE /etc/systemd/system/motlie-vfs-guest.service"
        --customize-hook="upload $BACKEND_ENV_FILE $MOTLIE_V15_BACKEND_ENV_PATH"
        --customize-hook="upload $DATASOURCE_CFG_FILE /etc/cloud/cloud.cfg.d/99_motlie_ch.cfg"
        --customize-hook="upload $AGENT_STATE_SETUP_FILE /usr/local/bin/motlie-agent-state-setup"
        --customize-hook="upload $AGENT_STATE_UNIT_FILE /etc/systemd/system/motlie-agent-state.service"
        --customize-hook="upload $SSH_BRIDGE_LOOP_FILE /usr/local/bin/motlie-vmm-vsock-ssh-loop"
        --customize-hook="upload $SSH_BRIDGE_UNIT_FILE /etc/systemd/system/motlie-vmm-vsock-ssh.service"
        --customize-hook="upload $EGRESS_SETUP_FILE /usr/local/bin/motlie-vmm-egress-setup"
        --customize-hook="upload $EGRESS_UNIT_FILE /etc/systemd/system/motlie-vmm-egress.service"
        --customize-hook="upload $OVERLAY_INIT_FILE /sbin/overlay-init"
        --customize-hook="upload $RUNTIME_CONTRACT_JSON /opt/motlie/v1.5/guest/guest-contract.json"
        --customize-hook="upload $TMUX_PROFILE /etc/profile.d/tmux-auto.sh"
        --customize-hook="upload $DOTENV_PROFILE /etc/profile.d/dotenv.sh"
        --customize-hook="upload $AGENT_PROFILE /etc/profile.d/agent-state.sh"
        --customize-hook="upload $APT_IPV4_CONF /etc/apt/apt.conf.d/99motlie-force-ipv4"
        --customize-hook="upload $MOTD_FILE /etc/motd"
        --customize-hook='chmod 755 "$1/usr/local/bin/motlie-agent-state-setup" "$1/usr/local/bin/motlie-vmm-vsock-ssh-loop" "$1/usr/local/bin/motlie-vmm-egress-setup" "$1/sbin/overlay-init"'
        --customize-hook='chmod 644 "$1/etc/profile.d/tmux-auto.sh" "$1/etc/profile.d/dotenv.sh" "$1/etc/profile.d/agent-state.sh" "$1/etc/apt/apt.conf.d/99motlie-force-ipv4" "$1/etc/motd"'
        --customize-hook='chroot "$1" systemctl enable motlie-vfs-guest.service'
        --customize-hook='chroot "$1" systemctl enable motlie-agent-state.service'
        --customize-hook='chroot "$1" systemctl enable motlie-vmm-vsock-ssh.service'
        --customize-hook='chroot "$1" systemctl enable motlie-vmm-egress.service'
        --customize-hook='cat > "$1/etc/fstab" << "FSTABEOF"
proc        /proc    proc    defaults        0      0
sysfs       /sys     sysfs   defaults        0      0
devtmpfs    /dev     devtmpfs defaults       0      0
FSTABEOF'
    )

    if [ "$INSTALL_AGENT_CLIS" = "1" ]; then
        ROOTFS_ARGS+=(
            --customize-hook='chroot "$1" npm install -g @openai/codex'
            --customize-hook='chroot "$1" npm install -g @anthropic-ai/claude-code'
        )
    fi

    ROOTFS_ARGS+=(
        --customize-hook='chroot "$1" apt-get clean'
        --customize-hook='rm -rf "$1/var/lib/apt/lists"/* "$1/var/lib/cloud"/* "$1/root/.npm"'
        --customize-hook='truncate -s 0 "$1/etc/machine-id"'
    )

    run_mmdebstrap "$BASE_ROOTFS" "${ROOTFS_ARGS[@]}"
else
    [ -f "$BASE_ROOTFS" ] || die "rootfs not found at $BASE_ROOTFS"
    echo "--- Step 3: Using existing rootfs: $BASE_ROOTFS ($(du -h "$BASE_ROOTFS" | cut -f1)) ---"
fi

write_guest_contract_json "$BASE_CONTRACT_JSON"
motlie_v15_require_guest_contract_json "$BASE_CONTRACT_JSON" "CH base artifacts"

if [ "$KERNEL_MODE" != "skip" ] || [ -f "$BASE_KERNEL" ]; then
    KERNEL_SIZE="$(du -h "$BASE_KERNEL" 2>/dev/null | cut -f1 || true)"
else
    KERNEL_SIZE="missing"
fi
ROOTFS_SIZE="$(du -h "$BASE_ROOTFS" | cut -f1)"

cat > "$ARTIFACTS_DIR/ch-build-result.json" <<EOF
{
  "ok": true,
  "backend": "ch",
  "contract": "$BASE_CONTRACT_JSON",
  "kernel": "$BASE_KERNEL",
  "rootfs": "$BASE_ROOTFS",
  "guest_binary": "$GUEST_BINARY_ABS",
  "build_git_sha": "$IMAGE_BUILD_GIT_SHA",
  "build_time_utc": "$IMAGE_BUILD_TIME_UTC"
}
EOF

echo ""
echo "=== CH artifact build complete ==="
ls -lh "$BASE_ARTIFACTS"
echo ""
echo "Size summary:"
echo "  kernel: $KERNEL_SIZE ($BASE_KERNEL)"
echo "  rootfs: $ROOTFS_SIZE ($BASE_ROOTFS)"
echo ""
echo "Next on a Linux CH host:"
echo "  cargo build -p motlie-vmm --example harness_v1_5"
echo "  ./target/debug/examples/harness_v1_5 --backend ch scenario ./libs/vmm/examples/v1.5/scenarios/multiguest-validate.json"
