# v1.2 Guest Image Notes

`v1.2` is the current, validated guest-image source of truth for the
`motlie-vnet` example / validation harness line.

The authoritative design/status document for this example is
[README.md](./README.md). Use that file first.

What is already true about the `v1.2` image build:

- it still builds one generic shared base image set under `artifacts/base/`
- guest identity remains launch-time state, not build-time state
- guest-local root writes still live in the per-launch writable ext4 overlay
- the image now includes:
  - `systemd-networkd`
  - the boot-time `motlie-vnet-egress` service that programs the current
    libslirp-compatible egress defaults
  - validation packages such as `curl`, `dnsutils`, and `ca-certificates`
  - `sudo`, `python3`, `npm`, and `bubblewrap`
  - baked Codex CLI and Claude Code CLI installs
  - boot-time symlink redirection of `~/.codex`, `~/.claude`, and
    `~/.config/claude-code` into the dedicated read/write `/agent-state` layer

Current runtime contract:

- the composed host flow is `cargo run -p motlie-vnet --example repl_host_v1_2`
- the launcher contract lives in [CH-HARNESS.md](./CH-HARNESS.md)
- the end-to-end validation runbook lives in [README.md](./README.md)

This file exists to describe the image contents and image-build methodology; it
is no longer a placeholder for future validation.
