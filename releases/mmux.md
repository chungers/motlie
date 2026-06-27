# mmux 0.1.1

`mmux` release notes for `2026-06-bright-beacon`.

## Summary

[DRAFT — needs David] `mmux` 0.1.1 updates Motlie's TUI tmux session selector while preserving the host-wide deployment model used by the prior release.

## Changes

[DRAFT — needs David] Adds visible stable session ids in list rows so operators can distinguish renamed or similarly named sessions.

[DRAFT — needs David] Adds list-pane quick search with `/` for case-insensitive substring matching.

[DRAFT — needs David] Adds list-pane sorting controls for name, tag grouping, host grouping, and return-to-activity sorting.

[DRAFT — needs David] Keeps endpoint labels derived from SSH endpoint identity, including user, host, non-default port, and non-default tmux socket.

[DRAFT — needs David] Keeps the package version visible through `mmux --version` and the Help modal.

## Install

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh | sh
/usr/local/bin/mmux --version
```

The installer defaults to `/usr/local/bin/mmux`. Use `--prefix` to select another prefix:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh
sh install-mmux.sh --prefix "$HOME/.local"
```

## Targets

| Target | Archive asset |
| --- | --- |
| linux-x64-musl | [motlie-mmux-v0.1.1-linux-x64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-linux-x64-musl.tar.gz) |
| linux-arm64-musl | [motlie-mmux-v0.1.1-linux-arm64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-linux-arm64-musl.tar.gz) |
| darwin-arm64 | [motlie-mmux-v0.1.1-darwin-arm64.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-darwin-arm64.tar.gz) |

Linux targets are static musl builds. The Darwin target is Apple Silicon only, ad-hoc signed, and verified from the installed path by the installer.

## Compatibility

[DRAFT — needs David] `mmux` still requires `tmux` on the target host.

The direct installer installs the native binary to `/usr/local/bin/mmux` by default.

[DRAFT — needs David] `/usr/local/bin/mmux` remains the recommended path for SSH `ForceCommand` deployments.

## Known Issues

[DRAFT — needs David] SSH `ForceCommand` integration tests remain environment-gated.

Archive payloads contain only `bin/mmux`.

## References

[`bins/mmux/docs/README.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/README.md) |
[`bins/mmux/docs/CLI.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/CLI.md) |
[`bins/mmux/docs/API.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/API.md) |
[`bins/mmux/docs/DESIGN.md`](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/bins/mmux/docs/DESIGN.md)
