# 2026-06-bright-beacon

Brownfield Motlie binary release for `mmux` 0.1.1 and `mstream` 0.1.0.

## Summary

This release updates `mmux`, the Motlie tmux session selector, and ships the
first release artifact set for `mstream`, the Motlie agent-facing workstream
orchestrator.

## Binaries

| Binary | Version | Channels | Targets |
| --- | --- | --- | --- |
| [mmux](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mmux.md) | 0.1.1 | archive, installer | linux-x64-musl, linux-arm64-musl, darwin-arm64 |
| [mstream](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mstream.md) | 0.1.0 | archive, installer | linux-x64-musl, linux-arm64-musl, darwin-arm64 |

## Install

```sh
# mmux
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh | sh

# mstream
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh | sh
```

Audit-before-run form:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh
shasum -a 256 install-mmux.sh
sh install-mmux.sh

curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh
shasum -a 256 install-mstream.sh
sh install-mstream.sh
```

## Changes

See per-binary notes:

- [mmux 0.1.1](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mmux.md)
- [mstream 0.1.0](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mstream.md)

## Verification

All archive assets ship with a single `SHA256SUMS` covering every archive in
this release. Verify with:

```sh
shasum -a 256 -c SHA256SUMS
```

Darwin binaries are Apple Silicon only and ad-hoc signed. Verify the installed
binary with:

```sh
codesign --verify --strict --verbose=2 <installed-path>
```

## Known Issues

Archive payloads contain `bin/<binary>` only; `README.md` and `LICENSE` files
are explicitly deferred for this release event.

## Assets

This release publishes the following assets at https://github.com/chungers/motlie/releases/tag/2026-06-bright-beacon :

- `motlie-mmux-v0.1.1-linux-x64-musl.tar.gz`
- `motlie-mmux-v0.1.1-linux-arm64-musl.tar.gz`
- `motlie-mmux-v0.1.1-darwin-arm64.tar.gz`
- `motlie-mstream-v0.1.0-linux-x64-musl.tar.gz`
- `motlie-mstream-v0.1.0-linux-arm64-musl.tar.gz`
- `motlie-mstream-v0.1.0-darwin-arm64.tar.gz`
- `SHA256SUMS`
- `install-mmux.sh`
- `install-mstream.sh`
- `manifest.toml` (workspace release manifest)
- `mmux.toml`, `mstream.toml` (per-binary release manifests)
- `notes.md`, `mmux.md`, `mstream.md` (release notes sources)
