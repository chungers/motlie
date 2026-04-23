# Skill Binaries

`voice-agent` installs platform-scoped binaries into this directory so the
voice skills can reuse optimized local builds without invoking `cargo run`.

Binary naming:

- `voice-agent-<os>-<arch>-<profile>-cuda`
- `voice-agent-<os>-<arch>-<profile>-cpu`

Example:

- `voice-agent-linux-aarch64-release-cuda`

Runtime selection:

- `release` is the default profile
- CUDA-ready hosts prefer `-cuda`
- CPU-only hosts prefer `-cpu`
- if the preferred flavor is missing, the wrapper falls back to the next best
  installed flavor before rebuilding
- legacy unsuffixed names such as `voice-agent-linux-aarch64-release` are still
  accepted as a fallback
- when a rebuild is needed, the wrapper prints a short `please wait` message for
  the human before building the optimized binary for that host
- if no shipped binary exists and the repo source tree is missing, the wrapper
  fails with a direct `source not available on this host` message
