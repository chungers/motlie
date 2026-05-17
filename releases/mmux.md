# mmux 0.1.0

Initial release of `mmux`, a TUI tmux session selector. Built as both an interactive
picker and a host-wide login entry point via SSH `ForceCommand`.

## Highlights

- **Live preview.** Highlighted session shows its active pane in real time; attach with a clean PTY handoff.
- **Multi-host.** `mmux ssh://a ssh://b ...` aggregates sessions across hosts into one activity-sorted list with a color legend. ([#235](https://github.com/chungers/motlie/issues/235))
- **Adaptive layout.** Auto-detects landscape vs portrait from PTY aspect ratio; portrait optimized for narrow terminals.
- **Session lifecycle.** Create, rename, tag, kill, and send keys from the picker.
- **`ForceCommand`-safe.** Static native binary in the SSH login path; no Node/shell runtime. ([#232](https://github.com/chungers/motlie/issues/232))
- **Script mode.** `mmux --script` prints the selected session for shell composition.

## Install

```sh
# Direct installer (recommended for host-wide / ForceCommand use)
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-05-apex-anchor/install-mmux.sh | sh

# Homebrew (macOS)
brew tap motlie/tap && brew install mmux

# npm — pick the package matching your platform
npm install -g @motlie/mmux-<linux-x64-musl|linux-arm64-musl|darwin-x64|darwin-arm64>
```

## Targets

`linux-x64-musl`, `linux-arm64-musl` (static, runs on Alpine), `darwin-x64`, `darwin-arm64` (ad-hoc signed).
Requires `tmux` on the target host.

## Known Issues

- SSH `ForceCommand` integration tests are env-gated, not yet in CI ([#232](https://github.com/chungers/motlie/issues/232)).
- Archive payloads contain `bin/mmux` only; `README.md` and `LICENSE` files are explicitly deferred for this release.

## References

[`docs/README.md`](https://github.com/chungers/motlie/blob/2026-05-apex-anchor/bins/mmux/docs/README.md) ·
[`docs/CLI.md`](https://github.com/chungers/motlie/blob/2026-05-apex-anchor/bins/mmux/docs/CLI.md) ·
[`docs/DESIGN.md`](https://github.com/chungers/motlie/blob/2026-05-apex-anchor/bins/mmux/docs/DESIGN.md) ·
parent feature [#226](https://github.com/chungers/motlie/issues/226)
