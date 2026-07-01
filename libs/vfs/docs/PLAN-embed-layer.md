# PLAN - Embedded Layer + Access Logging for motlie-vfs

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-07-01 | @codex-590-impl | Added implementation tracking for approved design #590 and marked completed tasks from the implementation pass. |
| 2026-07-01 | @codex-590-impl | Updated mstream consumer tasks for explicit `daemon start --mount <DIR>` and build-script-owned `MSTREAM_SKILLS_DIR`. |

## Scope

This plan tracks implementation of [DESIGN-embed-layer.md](./DESIGN-embed-layer.md) for issue #590. It is intentionally narrower than the historical VFS roadmap in [PLAN.md](./PLAN.md).

## Implementation Checklist

- [x] 1. Static VFS layer ([DESIGN FR-1..FR-4](./DESIGN-embed-layer.md#3-requirements))
  - [x] Add `LayerSource::Static` backed by `&'static include_dir::Dir<'static>`.
  - [x] Preserve union/shadow/whiteout/symlink merge behavior through `Layer` accessors.
  - [x] Publish static-only tags and expose static entries in layer/effective listings.
  - [x] Use zero-copy `Bytes::from_static` for static file reads.
  - [x] Use fixed-epoch static mtime and registration-time uid/gid.
  - [x] Reject static mutations as read-only; static xattr getters report no data.

- [x] 2. Server behavior ([DESIGN FR-3, FR-5](./DESIGN-embed-layer.md#3-requirements))
  - [x] Skip static layers in writable-layer selection.
  - [x] Propagate overlay mutation errors instead of discarding them.
  - [x] Return `EROFS` for writes/create/unlink/rmdir/rename/symlink attempts against static-only targets.
  - [x] Add `FsObserver` and `FsAccess` with path/bytes/errno/latency.
  - [x] Invoke observer after the mount `RwLock` is dropped.

- [x] 3. Feature split and local mount ([DESIGN §8](./DESIGN-embed-layer.md#8-logistics-re-anchored-for-plan))
  - [x] Add `fuser-client`, redefine `client = ["fuser-client", "vsock"]`, and add `local-mount`.
  - [x] Gate `FuseClient` on `fuser-client`; keep `guest::mount_all` gated on `client+vsock+linux`.
  - [x] Add Linux `client::local::mount_local()` using blocking `fuser::mount2()` with explicit unmount and stale-mount recovery.

- [x] 4. mstream first consumer ([DESIGN §4.8](./DESIGN-embed-layer.md#48-consumer-walkthrough-asset--mount-mapping-mstream))
  - [x] Bake `.agents/skills/project/` with `include_dir!("$MSTREAM_SKILLS_DIR")`, with `build.rs` computing the absolute source path.
  - [x] Emit `cargo:rustc-env=MSTREAM_SKILLS_DIR=<abs-skills-dir>` from `build.rs`.
  - [x] Add `build.rs` watches for the root, subdirectories, and files.
  - [x] Reject symlinks in the baked source tree during the mstream build.
  - [x] Depend on `motlie-vfs` with `features = ["local-mount"]`, not `client`.
  - [x] Mount only when `mstream daemon start --mount <DIR>` is provided.
  - [x] Degrade with a warning when `--mount` is omitted, the platform is unsupported, or FUSE mount is unavailable.
  - [x] Add CLI/build-env tests for `--mount`, omitted mount, and `MSTREAM_SKILLS_DIR`.
  - [x] Unmount during daemon graceful shutdown.

- [x] 5. Tests and verification ([DESIGN §5](./DESIGN-embed-layer.md#5-system-design--components-to-test))
  - [x] Add static-layer unit tests for resolve, readdir, tag publication, shadow/whiteout, symlink semantics, and read-only mutation rejection.
  - [x] Add server tests for static read, fixed mtime, and `EROFS` on write/create/unlink.
  - [x] Add observer tests for fh-based read/release paths, byte counts, errno, and post-lock callback scope.
  - [x] Add mstream embedded asset tests.
  - [x] Verify `motlie-vfs` default, `local-mount`, and `client` feature builds.

## Known Gaps

- [ ] @codex-590-impl 2026-07-01 -- Generic library registration cannot independently reject symlinks from `include_dir::Dir` alone because that runtime value exposes baked files and directories, not source-tree symlink provenance. mstream enforces the approved bake-time rejection by scanning the source tree in `build.rs`; a future generic API should accept a source path or manifest if arbitrary consumers need the same guarantee.
- [ ] @codex-590-impl 2026-07-01 -- `include_dir` 0.7 requires a string-literal macro argument, so `skills.rs` uses `include_dir!("$MSTREAM_SKILLS_DIR")` instead of the requested `include_dir!(env!("MSTREAM_SKILLS_DIR"))`. `build.rs` still owns the absolute path and exposes it through `MSTREAM_SKILLS_DIR`.
- [ ] Live FUSE mount testing still requires host `/dev/fuse`/`fusermount3` access. Unit and compile gates cover the degrade path and mount helper construction; a live mount smoke should be run on a FUSE-enabled Linux host before depending operationally on the mount.
