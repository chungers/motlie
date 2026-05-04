#!/bin/sh
set -eu

# MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V3
# This script is immutable base-image content for v1.5 Vz. It must not chown
# VFS-backed `/agent-state/*` or `/home/*` paths: ownership is presented by the
# VFS layer for the active guest uid/gid, and chown is not a valid readiness
# operation for first-contact SSH.

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"
    config_dir="$home_dir/.config"
    codex_dst="$home_dir/.codex"
    claude_dst="$home_dir/.claude"
    claude_code_dst="$config_dir/claude-code"

    [ -d "$home_dir" ] || return 0

    for mount_path in "$codex_dst" "$claude_dst" "$claude_code_dst"; do
        umount "$mount_path" >/dev/null 2>&1 || true
    done

    install -d -m 0755 "$config_dir"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    rm -rf "$codex_dst" "$claude_dst" "$claude_code_dst"
    install -d -m 0700 "$codex_dst" "$claude_dst" "$claude_code_dst"

    mount --bind /agent-state/codex "$codex_dst"
    mount --bind /agent-state/claude "$claude_dst"
    mount --bind /agent-state/claude-code "$claude_code_dst"
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
