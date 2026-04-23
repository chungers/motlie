# Skill Binaries

`voice-agent` installs platform-scoped binaries into this directory so the
voice skills can reuse optimized local builds without invoking `cargo run`.

Binary naming:

- `voice-agent-<os>-<arch>-release`
- `voice-agent-<os>-<arch>-debug`

Example:

- `voice-agent-linux-aarch64-release`
