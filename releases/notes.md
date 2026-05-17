# 2026-05-apex-anchor

First Motlie release event. Ships `mmux` 0.1.0.

## Summary

First Motlie release event. Ships `mmux` 0.1.0 — a TUI tmux session selector with
live active-pane preview, multi-host SSH aggregation, and host-wide `ForceCommand`-safe
operation. See [`mmux.md`](mmux.md) for capabilities, install, and references.

## Binaries

| Binary | Version | Channels | Targets |
|---|---|---|---|
| [mmux](mmux.md) | 0.1.0 | archive, installer, npm, homebrew | linux-x64-musl, linux-arm64-musl, darwin-x64, darwin-arm64 |

## Install

Each binary has its own install guidance. See per-binary notes:

- [mmux 0.1.0](mmux.md) — TUI tmux session selector.

## Changes

See per-binary notes for binary-specific changes.

## Verification

All archive assets ship with a single `SHA256SUMS` covering every archive in this release. Verify with:

```sh
shasum -a 256 -c SHA256SUMS
```

Darwin binaries are ad-hoc signed. Verify the installed binary with:

```sh
codesign --verify --strict --verbose=2 <installed-path>
```

## Known Issues

See per-binary notes.

## Assets

This release publishes the following assets at https://github.com/chungers/motlie/releases/tag/2026-05-apex-anchor :

- `motlie-mmux-v0.1.0-linux-x64-musl.tar.gz`
- `motlie-mmux-v0.1.0-linux-arm64-musl.tar.gz`
- `motlie-mmux-v0.1.0-darwin-x64.tar.gz`
- `motlie-mmux-v0.1.0-darwin-arm64.tar.gz`
- `SHA256SUMS`
- `install-mmux.sh`
- `manifest.toml` (workspace release manifest)
- `mmux.toml` (per-binary release manifest)
- `notes.md`, `mmux.md` (release notes sources)

## Per-Binary Notes

### mmux 0.1.0

See [`mmux.md`](mmux.md).
