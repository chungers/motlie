# motlie-tmux SFTP Proposal

## Status: Draft

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-14 | @codex | Refined design per user decisions: greenfield/breaking changes accepted, API renamed to `upload` / `download`, overwrite semantics made configurable, directory support included now, and v1/file-only phasing removed. |
| 2026-03-14 | @codex | Initial SFTP design note: add transport/host-level file transfer to complement transport `exec()`, use SFTP under the existing `russh` connection, and keep tmux pane `Target::exec()` separate. |

## Purpose

This note outlines the changes needed to add host-level file transfer to
`libs/tmux` without conflating it with tmux-pane command execution.

The current library has two distinct execution layers:

- `TransportKind::exec()` runs shell commands on the host, locally or over SSH.
- `Target::exec()` runs a shell command inside a tmux pane and extracts output via
  sentinel polling.

File transfer belongs beside transport `exec()`, not beside pane `Target::exec()`.

## Recorded Decisions

- This is **greenfield** work. Breaking API changes are acceptable; there is no
  migration or backwards-compatibility requirement for SFTP-related APIs.
- Use **SFTP**, not the SCP protocol.
- Use **`upload` / `download`** naming in the public API for clarity.
- Support **files and directories now**, not as a later phase.
- Overwrite behavior is **configurable**. If overwrite is disabled and the
  destination already exists, return `Err`.

## Recommendation

Use SFTP under the existing `russh` connection.

Reasons:

- The SSH transport already owns a live authenticated `russh` connection in
  `src/transport.rs`.
- SFTP is structured and binary-safe; it avoids shell quoting and ad hoc parsing.
- Implementing literal SCP would either require shelling out to system `scp` or
  implementing the SCP protocol over an exec channel.
- SFTP fits the current in-process architecture and is easier to test across
  `Local`, `Mock`, and `Ssh` transports.

## Scope

### Goals

- Add host-level upload and download support that complements transport `exec()`.
- Support both regular files and directories.
- Keep the API transport-agnostic across `Local`, `Mock`, and `Ssh`.
- Keep failure modes non-panicking and explicit.

### Non-goals

- Literal SCP protocol compatibility
- Full rsync-style synchronization or delta transfer
- Progress callbacks
- Automatic migration compatibility for earlier experimental transfer APIs
- Permission/ownership preservation as a required contract

## Current Architecture Constraints

Relevant current boundaries:

- `TransportKind` exposes the transport layer in `src/transport.rs`.
- `SshTransport` already stores a persistent authenticated
  `russh::client::Handle<SshHandler>`.
- `HostHandle` is the transport-agnostic public facade in `src/host.rs`.
- `Target::exec()` is tmux-specific and should remain tmux-specific.

This means file transfer should be introduced as a transport capability and then
forwarded through `HostHandle`.

## Proposed Public API

Use upload/download terminology at the host layer:

```rust
pub struct TransferOptions {
    pub overwrite: bool,
    pub recursive: bool,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            overwrite: true,
            recursive: false,
        }
    }
}

impl HostHandle {
    pub async fn upload(
        &self,
        local_path: &std::path::Path,
        remote_path: &str,
        opts: &TransferOptions,
    ) -> Result<()>;

    pub async fn download(
        &self,
        remote_path: &str,
        local_path: &std::path::Path,
        opts: &TransferOptions,
    ) -> Result<()>;
}
```

And at the transport layer:

```rust
impl TransportKind {
    pub async fn upload(
        &self,
        local_path: &std::path::Path,
        remote_path: &str,
        opts: &TransferOptions,
    ) -> Result<()>;

    pub async fn download(
        &self,
        remote_path: &str,
        local_path: &std::path::Path,
        opts: &TransferOptions,
    ) -> Result<()>;
}
```

### Directory semantics

- If the source is a directory, `opts.recursive` must be `true`, otherwise return `Err`.
- A recursive transfer copies the directory tree rooted at the source.
- The destination root may be created as part of the transfer, but missing parent
  directories above that root should still return `Err`.
- If `opts.overwrite == false` and the destination exists, return `Err`.

## Proposed Internal Changes

### 1. Dependencies

Update `libs/tmux/Cargo.toml`:

- Add explicit `russh-sftp` dependency.

Even though `Cargo.lock` already contains `russh-sftp` transitively, the crate
should not rely on a transitive dependency for direct use.

### 2. Types and API Surface

Add `TransferOptions` as a public type. This captures the decisions that must be
configurable at the API boundary today:

- `overwrite`
- `recursive`

No migration layer is needed because this is greenfield work.

### 3. Transport Layer

Extend `src/transport.rs`:

- Add `TransportKind::upload()` and `TransportKind::download()`
- Add matching private implementations on:
  - `LocalTransport`
  - `MockTransport`
  - `SshTransport`

Recommended behavior by transport:

- `LocalTransport`
  - Implement upload/download as local filesystem copy operations
  - Support both files and recursive directories
  - Wrap each top-level transfer in the existing transport timeout
- `MockTransport`
  - Add an in-memory filesystem/tree model
  - Support deterministic transfer error injection for tests
- `SshTransport`
  - Use SFTP for file and directory operations
  - Reuse the existing authenticated SSH handle to open SFTP channels
  - Bound each top-level transfer by `SshConfig::timeout`

Opening a fresh SFTP channel per top-level transfer remains the simplest starting
point. Shared-client optimization can be added later if needed.

### 4. Host Layer

Extend `src/host.rs`:

- Add public `HostHandle::upload()` / `download()` wrappers
- Forward directly to `self.inner.transport`

Do **not** add transfer methods on `Target`.

Reason: `Target` addresses tmux session/window/pane objects, while upload/download
address the host filesystem. Mixing them would blur two different abstractions.

### 5. Public Exports

Update `src/lib.rs` to export `TransferOptions` and any other public transfer types
added during implementation.

### 6. Errors and Validation

Keep failure modes non-panicking:

- missing source -> `Err(...)`
- destination exists with `overwrite=false` -> `Err(...)`
- directory source with `recursive=false` -> `Err(...)`
- permission denied -> `Err(...)`
- SSH/SFTP subsystem failure -> `Err(...)`
- timeout -> `Err(...)`

Do not use `assert!` / `expect()` as guards for public transfer inputs.

## Semantics

- Binary-safe: file contents are copied as raw bytes
- File and directory transfers are both supported
- Recursive directory transfer requires `opts.recursive=true`
- Overwrite behavior is controlled by `opts.overwrite`
- Timeout semantics match existing transports:
  - `LocalTransport::timeout` bounds each top-level transfer
  - `SshConfig::timeout` bounds each top-level SFTP transfer

## Implementation Sketch

At a high level:

1. `HostHandle::upload(local, remote, opts)` forwards to `TransportKind::upload`.
2. `TransportKind::upload` dispatches to the concrete transport.
3. `SshTransport::upload` opens an SFTP channel on the existing SSH connection and
   recursively copies either a file or a directory tree according to `TransferOptions`.

The download path is symmetric.

This keeps transfer support orthogonal to:

- tmux discovery
- tmux control
- pane capture
- pane `Target::exec()`

## Testing Plan

### Unit Tests

- `MockTransport` file upload/download round-trip
- `MockTransport` directory upload/download round-trip
- `MockTransport` overwrite=false and recursive=false error paths
- `TransportKind` dispatch tests for new methods

### Local Integration Tests

- file upload/download round-trip to temp paths
- directory upload/download round-trip for a nested tree
- overwrite=false conflict path
- recursive=false directory rejection
- missing-path and permission-denied error paths

### SSH Integration Tests

Add env-gated integration tests similar to the existing SSH transport tests:

- upload a file to a temp remote path, then download it back
- upload a directory tree recursively, then download it back
- verify overwrite=false and recursive=false behavior

These tests should require the same SSH prerequisites already documented for
`SshTransport`.

## Documentation Changes Needed When Implemented

- Add a new section to `docs/API.md` for host-level upload/download
- Clarify the distinction between:
  - transport `exec()`
  - host upload/download
  - tmux-pane `Target::exec()`
- Add at least one runnable example under `examples/` once implementation starts

## Remaining Clarifications

The major product decisions are now fixed. The remaining implementation-level
clarifications are narrower:

1. How much metadata, if any, should be preserved during directory transfers?
2. Should symlinks be rejected initially or copied as links when encountered?
3. Do we want a transfer result/report type later, or keep `Result<()>` initially?

## Summary

The clean design is:

- implement **SFTP-backed** host upload/download
- place it on the **transport** boundary
- expose it through **HostHandle**
- keep it **out of `Target`**
- support **files and directories now**
- make overwrite behavior **configurable**

That complements the existing transport `exec()` API without weakening the tmux
abstractions already in place.
