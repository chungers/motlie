# mmux 0.1.0

## Summary

First release of `mmux`, a terminal UI tmux session selector. Pick from active tmux sessions with an ncurses-style chooser; attach with a keypress.

## Changes

(Draft — to be filled in by release owner before final tag. Skill should not invent claims from commit subjects.)

## Install

### Direct installer (recommended for host-wide / SSH `ForceCommand` use)

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-05-apex-anchor/install-mmux.sh | sh
mmux --help
```

The installer detects OS + architecture + libc and installs the matching static binary to `/usr/local/bin/mmux`.

### Homebrew (macOS)

```sh
brew tap motlie/tap
brew install mmux
mmux --help
```

### npm (developer / CI install)

```sh
# Pick the package matching your platform:
npm install -g @motlie/mmux-linux-x64-musl    # Linux x64
npm install -g @motlie/mmux-linux-arm64-musl  # Linux arm64
npm install -g @motlie/mmux-darwin-x64        # macOS Intel
npm install -g @motlie/mmux-darwin-arm64      # macOS Apple Silicon
mmux --help
```

### Direct download

Download the platform tarball from the release page, extract, and install `bin/mmux` to a directory on PATH.

```sh
curl -fsSL -o mmux.tar.gz https://github.com/chungers/motlie/releases/download/2026-05-apex-anchor/motlie-mmux-v0.1.0-<target>.tar.gz
tar xzf mmux.tar.gz
sudo install -m 755 bin/mmux /usr/local/bin/mmux
mmux --help
```

## Targets

| OS | Arch | Libc / Link | Archive | npm package |
|---|---|---|---|---|
| linux | x64 | musl / static | `motlie-mmux-v0.1.0-linux-x64-musl.tar.gz` | `@motlie/mmux-linux-x64-musl` |
| linux | arm64 | musl / static | `motlie-mmux-v0.1.0-linux-arm64-musl.tar.gz` | `@motlie/mmux-linux-arm64-musl` |
| darwin | x64 | native | `motlie-mmux-v0.1.0-darwin-x64.tar.gz` | `@motlie/mmux-darwin-x64` |
| darwin | arm64 | native | `motlie-mmux-v0.1.0-darwin-arm64.tar.gz` | `@motlie/mmux-darwin-arm64` |

## Compatibility

- Requires `tmux` installed on the host (via system package manager).
- Linux: static musl binaries — runs on Alpine, glibc distros, and minimal containers without additional runtime dependencies.
- macOS: ad-hoc signed (`codesign --sign -`). Apple Silicon hosts validate signature at startup; the installer re-signs the installed copy.
- SSH `ForceCommand` safe: `mmux` is a native binary with no Node/JS runtime in its launch path. `Match Group mmux-users` + `ForceCommand /usr/local/bin/mmux` is supported.

## Verification

```sh
# Checksum
shasum -a 256 -c SHA256SUMS

# macOS signature
codesign --verify --strict --verbose=2 /usr/local/bin/mmux
```

## Known Issues

(Draft — release owner to fill in before final tag.)
