# mmux 0.1.1

`mmux` is Motlie's TUI tmux session selector. It lists tmux sessions, live
previews the highlighted session's active pane, and attaches the user's
terminal to the selected session with a clean PTY handoff.

## Summary

This brownfield update keeps the host-wide `ForceCommand`-safe deployment model
from `mmux` 0.1.0 and adds the post-apex usability work now present on `main`.

## Changes

- Adds visible stable session ids in list rows so operators can distinguish
  renamed or similarly named sessions.
- Adds list-pane quick search with `/` for case-insensitive substring matching.
- Adds list-pane sorting controls for name, tag grouping, host grouping, and
  return-to-activity sorting.
- Keeps endpoint labels derived from SSH endpoint identity, including user,
  host, non-default port, and non-default tmux socket.
- Keeps the package version visible through `mmux --version` and the Help modal.

## Install

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh | sh
/usr/local/bin/mmux --version
```

## Targets

`linux-x64-musl`, `linux-arm64-musl`, and `darwin-arm64`.
Linux targets are static musl builds. The Darwin target is Apple Silicon only, ad-hoc signed, and
verified from the installed path.

## Compatibility

`mmux` still requires `tmux` on the target host. The direct installer installs
the native binary to `/usr/local/bin/mmux` by default, which remains the
recommended path for SSH `ForceCommand` deployments.

## Known Issues

- SSH `ForceCommand` integration tests remain environment-gated.
- Archive payloads contain `bin/mmux` only; `README.md` and `LICENSE` files are
  explicitly deferred for this release event.

## References

[`docs/README.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/README.md) |
[`docs/CLI.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/CLI.md) |
[`docs/API.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/API.md) |
[`docs/DESIGN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/DESIGN.md)
