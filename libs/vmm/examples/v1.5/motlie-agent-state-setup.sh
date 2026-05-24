#!/bin/sh
set -eu

# MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V4
# This script is immutable base-image content for v1.5 CH/VZ. It must not chown
# VFS-backed `/agent-state/*` or `/home/*` paths: ownership is presented by the
# VFS layer for the active guest uid/gid, and chown is not a valid readiness
# operation for first-contact SSH. Use symlinks instead of nested bind mounts so
# the contract works across FUSE-backed CH/VZ home directories.

is_mounted() {
    mount_path="$1"
    awk -v mount_path="$mount_path" '$2 == mount_path { found = 1 } END { exit found ? 0 : 1 }' /proc/mounts
}

wait_for_mount() {
    mount_path="$1"
    for _attempt in $(seq 1 120); do
        if is_mounted "$mount_path"; then
            return 0
        fi
        sleep 1
    done
    echo "motlie-agent-state-setup: timed out waiting for $mount_path mount" >&2
    return 1
}

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"
    config_dir="$home_dir/.config"
    codex_dst="$home_dir/.codex"
    claude_dst="$home_dir/.claude"
    claude_code_dst="$config_dir/claude-code"

    [ -d "$home_dir" ] || return 0
    # CH systemd and VZ OpenRC can start this helper after the VFS guest process
    # is running but before FUSE has mounted each path. Wait here so agent state
    # is never written under a future mount point and hidden by the VFS layer.
    wait_for_mount /agent-state
    wait_for_mount "$home_dir"

    install -d -m 0755 "$config_dir"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    rm -rf "$codex_dst" "$claude_dst" "$claude_code_dst"
    ln -s /agent-state/codex "$codex_dst"
    ln -s /agent-state/claude "$claude_dst"
    ln -s /agent-state/claude-code "$claude_code_dst"
}

for home_dir in /home/*; do
    [ -d "$home_dir" ] || continue
    user_name="$(basename "$home_dir")"
    case "$user_name" in
        admin|ubuntu) continue ;;
    esac
    if id -u "$user_name" >/dev/null 2>&1; then
        setup_user "$user_name"
    fi
done
