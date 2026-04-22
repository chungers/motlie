#!/bin/sh
set -eu

setup_user() {
    user_name="$1"
    home_dir="/home/$user_name"

    [ -d "$home_dir" ] || return 0

    install -d -m 0755 "$home_dir/.config"
    install -d -m 0700 /agent-state/codex /agent-state/claude /agent-state/claude-code /agent-state/codex/sqlite

    chown -R "$user_name:$user_name" \
        "$home_dir/.config" \
        /agent-state/codex \
        /agent-state/codex/sqlite \
        /agent-state/claude \
        /agent-state/claude-code || true

    rm -rf "$home_dir/.codex" "$home_dir/.claude" "$home_dir/.config/claude-code"
    ln -sfn /agent-state/codex "$home_dir/.codex"
    ln -sfn /agent-state/claude "$home_dir/.claude"
    ln -sfn /agent-state/claude-code "$home_dir/.config/claude-code"
    chown -h "$user_name:$user_name" \
        "$home_dir/.codex" \
        "$home_dir/.claude" \
        "$home_dir/.config/claude-code" || true
}

for user_name in alice bob; do
    if id -u "$user_name" >/dev/null 2>&1; then
        setup_user "$user_name"
    fi
done
