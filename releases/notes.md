# 2026-06-bright-beacon

Brownfield Motlie binary release for `mmux` 0.1.1 and `mstream` 0.1.0.

## Summary

This release refreshes `mmux`, Motlie's tmux session selector, and introduces the first release artifact set for `mstream`, Motlie's agent-facing workstream orchestrator.

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
sed -n '1,220p' install-mmux.sh
sh install-mmux.sh

curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh
sed -n '1,220p' install-mstream.sh
sh install-mstream.sh
```

The installer scripts select the platform-matched archive, download `SHA256SUMS` from https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/SHA256SUMS, verify the selected archive checksum, extract `bin/<binary>`, install to `/usr/local/bin` by default, and verify `<binary> --version`.

## Changes

`mmux` 0.1.1 carries the post-apex usability work described in the per-binary notes.

`mstream` 0.1.0 packages the implemented CLI and daemon surface for direct archive and installer distribution.

See per-binary notes:

- [mmux 0.1.1](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mmux.md)
- [mstream 0.1.0](https://github.com/chungers/motlie/blob/2026-06-bright-beacon/releases/mstream.md)

## Verification

Linux archives are static musl builds for `linux-x64-musl` and `linux-arm64-musl`.

Darwin archives are Apple Silicon `darwin-arm64` builds. The Darwin binaries are ad-hoc signed and the installer re-signs the installed binary before verifying it from the installed path.

To verify an installed Darwin binary manually:

```sh
codesign --verify --strict --verbose=2 <installed-path>
```

## Known Issues

Known issues are listed in the per-binary notes.

Archive payloads contain only the executable at `bin/<binary>`.

## Assets

Archive assets:

| Binary | Target | Asset |
| --- | --- | --- |
| mmux | linux-x64-musl | [motlie-mmux-v0.1.1-linux-x64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-linux-x64-musl.tar.gz) |
| mmux | linux-arm64-musl | [motlie-mmux-v0.1.1-linux-arm64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-linux-arm64-musl.tar.gz) |
| mmux | darwin-arm64 | [motlie-mmux-v0.1.1-darwin-arm64.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mmux-v0.1.1-darwin-arm64.tar.gz) |
| mstream | linux-x64-musl | [motlie-mstream-v0.1.0-linux-x64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-linux-x64-musl.tar.gz) |
| mstream | linux-arm64-musl | [motlie-mstream-v0.1.0-linux-arm64-musl.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-linux-arm64-musl.tar.gz) |
| mstream | darwin-arm64 | [motlie-mstream-v0.1.0-darwin-arm64.tar.gz](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/motlie-mstream-v0.1.0-darwin-arm64.tar.gz) |

Installer and checksum assets used by the install commands:

- [install-mmux.sh](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mmux.sh)
- [install-mstream.sh](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/install-mstream.sh)
- [SHA256SUMS](https://github.com/chungers/motlie/releases/download/2026-06-bright-beacon/SHA256SUMS)
