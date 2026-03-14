# motlie-tmux SFTP Proposal

## Status: Draft

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-14 | @codex | Initial SFTP design note: add transport/host-level file transfer to complement transport `exec()`, use SFTP under the existing `russh` connection, and keep tmux pane `Target::exec()` separate. |

## Purpose

This note outlines the changes needed to add remote file transfer to `libs/tmux`
without conflating it with tmux-pane command execution.

The current library has two distinct execution layers:

- `TransportKind::exec()` runs shell commands on the host, locally or over SSH.
- `Target::exec()` runs a shell command inside a tmux pane and extracts output via
  sentinel polling.

File transfer belongs beside transport `exec()`, not beside pane `Target::exec()`.

## Recommendation

Use **SFTP**, not the SCP protocol.

Reasons:

- The SSH transport already owns a live authenticated `russh` connection in
  `src/transport.rs`.
- SFTP is structured and binary-safe; it avoids shell quoting and ad hoc parsing.
- Implementing literal SCP would either require shelling out to system `scp` or
  implementing the SCP protocol over an exec channel.
- SFTP fits the current in-process architecture and is easier to test across
  `Local`, `Mock`, and `Ssh` transports.

If user-facing docs want to describe this as “scp-like remote copy”, keep the
implementation note honest: it is SFTP-backed.

## Scope

### Goals

- Add host-level file read/write support that complements transport `exec()`.
- Keep the API transport-agnostic across `Local`, `Mock`, and `Ssh`.
- Keep v1 simple and robust: files only, binary-safe, no panics as validation.

### Non-goals

- Literal SCP protocol compatibility
- Recursive directory copy
- Full directory sync / rsync semantics
- Progress callbacks
- Permission/ownership preservation in v1
- Streaming APIs in v1

## Current Architecture Constraints

Relevant current boundaries:

- `TransportKind` exposes only `exec()`, `is_healthy()`, and `open_shell()` in
  `src/transport.rs`.
- `SshTransport` already stores a persistent authenticated
  `russh::client::Handle<SshHandler>`.
- `HostHandle` is the transport-agnostic public facade in `src/host.rs`.
- `Target::exec()` is tmux-specific and should remain tmux-specific.

This means file transfer should be introduced as a transport capability and then
forwarded through `HostHandle`.

## Proposed Public API

Prefer transport-neutral naming over protocol naming:

```rust
impl HostHandle {
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>>;
    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<()>;
}
```

And at the transport layer:

```rust
impl TransportKind {
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>>;
    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<()>;
}
```

Why `read_file` / `write_file` instead of `scp_to` / `scp_from`:

- They map naturally to `LocalTransport`, where “upload” and “download” are
  awkward.
- They describe semantics instead of protocol.
- A higher-level copy helper can be added later without changing the transport
  primitive.

If overwrite behavior needs to be explicit in v1, add a small options type:

```rust
pub enum WriteMode {
    Overwrite,
    CreateNew,
}
```

Then:

```rust
pub async fn write_file(&self, path: &str, data: &[u8], mode: WriteMode) -> Result<()>;
```

For maximum simplicity, `Overwrite` as the default is acceptable if documented
clearly.

## Proposed Internal Changes

### 1. Dependencies

Update `libs/tmux/Cargo.toml`:

- Add explicit `russh-sftp` dependency.

Even though `Cargo.lock` already contains `russh-sftp` transitively, the crate
should not rely on a transitive dependency for direct use.

### 2. Transport Layer

Extend `src/transport.rs`:

- Add `TransportKind::read_file()` and `TransportKind::write_file()`
- Add matching private implementations on:
  - `LocalTransport`
  - `MockTransport`
  - `SshTransport`

Recommended behavior by transport:

- `LocalTransport`
  - Use `tokio::fs::read` / `tokio::fs::write`
  - Wrap each operation in the existing transport timeout
- `MockTransport`
  - Add in-memory file storage keyed by path
  - Support deterministic success/error injection for tests
- `SshTransport`
  - Open a fresh SFTP subsystem/channel per operation
  - Reuse the existing authenticated SSH handle only to open the channel
  - Bound the full operation by `SshConfig::timeout`

Opening a fresh SFTP channel per operation is the simplest and most consistent
starting point. Caching an SFTP client can be revisited later if performance
actually matters.

### 3. Host Layer

Extend `src/host.rs`:

- Add public `HostHandle::read_file()` / `write_file()` wrappers
- Forward directly to `self.inner.transport`

Do **not** add file transfer methods on `Target`.

Reason: `Target` addresses tmux session/window/pane objects, while file transfer
addresses the host filesystem. Mixing them would blur two different abstractions.

### 4. Public Exports

Update `src/lib.rs` if new transfer-related types are introduced, for example:

- `WriteMode`
- future metadata types if added later

### 5. Errors and Validation

Keep failure modes non-panicking:

- missing file -> `Err(...)`
- permission denied -> `Err(...)`
- SSH/SFTP subsystem failure -> `Err(...)`
- timeout -> `Err(...)`

Do not use `assert!` / `expect()` as guards for public transfer inputs.

## Recommended Semantics for v1

- Binary-safe: file contents are raw bytes (`Vec<u8>`)
- Files only: no recursive copy
- No implicit parent-directory creation
- No implicit chmod/chown preservation
- Timeout semantics match existing transports:
  - `LocalTransport::timeout` bounds each file operation
  - `SshConfig::timeout` bounds each SFTP operation

The absence of implicit parent creation is intentional for robustness. If the
destination directory does not exist, return an error instead of guessing.

## Implementation Sketch

At a high level:

1. `HostHandle::write_file(path, data)` forwards to `TransportKind::write_file`.
2. `TransportKind::write_file` dispatches to the concrete transport.
3. `SshTransport::write_file` opens an SFTP channel on the existing SSH
   connection, writes bytes, closes the handle, and returns `Result<()>`.

The read path is symmetric.

This keeps transfer support orthogonal to:

- tmux discovery
- tmux control
- pane capture
- pane `Target::exec()`

## Testing Plan

### Unit Tests

- `MockTransport` read/write round-trip
- `MockTransport` injected read/write failures
- `TransportKind` dispatch tests for new methods

### Local Integration Tests

- `HostHandle::write_file()` writes bytes to a temp file
- `HostHandle::read_file()` reads the same bytes back
- binary payload round-trip
- missing-path and permission-denied error paths

### SSH Integration Tests

Add env-gated integration tests similar to the existing SSH transport tests:

- upload bytes to a temp path on the SSH target
- read them back
- verify timeout/error behavior on invalid paths

These tests should require the same SSH prerequisites already documented for
`SshTransport`.

## Documentation Changes Needed When Implemented

- Add a new section to `docs/API.md` for host-level file transfer
- Clarify the distinction between:
  - transport `exec()`
  - host file transfer
  - tmux-pane `Target::exec()`
- Add at least one runnable example under `examples/` once implementation starts

## Open Questions

1. Should `write_file()` overwrite by default, or require explicit `WriteMode`?
2. Do we want path-only primitives (`read_file` / `write_file`) first, or also
   a convenience host-local copy helper?
3. Should future directory-oriented operations live in the same API, or behind a
   separate module once they exist?

## Summary

The clean design is:

- implement **SFTP-backed** file transfer
- place it on the **transport** boundary
- expose it through **HostHandle**
- keep it **out of `Target`**
- start with **binary-safe whole-file read/write**

That complements the existing transport `exec()` API without weakening the tmux
abstractions already in place.
