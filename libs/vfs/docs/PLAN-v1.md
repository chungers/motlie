# motlie-vfs v1 Delivery Plan

Extracted from [PLAN.md](./PLAN.md) — v1-only phases and tasks with v1.5/v2 scope excluded.
See [DESIGN.md](./DESIGN.md) for full architectural context and requirements.

## Status

- Phases 1.1–1.3, 2.1–2.2, 4.1: **complete** (PRs #118, #120)
- Phase 2.3: not started
- Phase 4.2: not started — **critical path** for real FUSE mount
- Phase 5.1 (v1 subset): not started — operational harness

76 tests passing (64 unit + 12 transport integration), 0 warnings.

---

## 1. Foundation — complete

### 1.1 Phase 0: Bootstrap Crate and Workspace

Design references: [Architectural Layering and Roadmap](./DESIGN.md), [Crate Structure](./DESIGN.md), [Feature Flags](./DESIGN.md)

- [x] 1.1.1 Add `libs/vfs` to workspace members.
- [x] 1.1.2 Create `libs/vfs/Cargo.toml` with the feature graph described in the DESIGN.
- [x] 1.1.3 Add initial dependency set for core, vsock, client, and example PoC modules.
- [x] 1.1.4 Create public module skeletons with feature gating only, no behavior yet.
- [x] 1.1.5 Ensure `cargo check -p motlie-vfs` succeeds with default features.
- [x] 1.1.6a Add `libs/vfs/examples/simple_host.rs` and `libs/vfs/examples/README.md` scaffolding.
- [x] 1.1.6b Add `libs/vfs/src/client/guest.rs` scaffolding.
- [x] 1.1.6c Add `libs/vfs/bins/motlie-vfs-guest.rs` scaffolding.
- [x] 1.1.7 Run `cargo check -p motlie-vfs`.
- [x] 1.1.8 Run a workspace `cargo check`.

### 1.2 Phase 1: Core Types and Error Surface

Design references: [Core Types: FsOp and FsResult](./DESIGN.md), [API Design](./DESIGN.md)

- [x] 1.2.1 Implement `FsOp`, `FsResult`, `FileAttr`, `DirEntry`, `FsStats`, and `SetAttrFields`.
- [x] 1.2.2 Implement `FsEvent` and `FsOpKind` (with `from_op()` conversion).
- [x] 1.2.3 Implement `PolicyFn` and `AllowAll`.
- [x] 1.2.4 Use `anyhow::Result` for operational APIs, matching the workspace convention.
- [x] 1.2.5 Add serde coverage for all wire-visible types.
- [x] 1.2.6 Ensure the core `FileAttr`/`SetAttrFields` model carries `uid`, `gid`, and `mode`.
- [x] 1.2.7 Add serde round-trip tests for `FsOp` and `FsResult`.
- [x] 1.2.8 Add allow/deny tests for `PolicyFn`.
- [x] 1.2.9 Add serde coverage tests for synthetic attrs carrying `uid`, `gid`, and `mode`.

### 1.3 Phase 2: InodeTable and Path Model

Design references: [Ordered Layer-Stack Model](./DESIGN.md), [Inode Management](./DESIGN.md), [Generation Semantics](./DESIGN.md)

- [x] 1.3.1 Implement per-mount `InodeTable`.
- [x] 1.3.2 Support shared inode namespace across disk entries, `Content`, `SyntheticDir`, and whiteout bookkeeping.
- [x] 1.3.3 Implement inode allocation rules for each entry kind.
- [x] 1.3.4 Implement generation tracking rules from the DESIGN.
- [x] 1.3.5 Implement path-to-inode reverse lookup.
- [x] 1.3.6 Implement stable-within-mount-lifetime behavior.
- [x] 1.3.7 Ensure inode data structures do not require every non-overlay inode to carry a concrete `host_path`.
- [x] 1.3.8 Add allocate/lookup/reuse tests for disk entries.
- [x] 1.3.9 Add allocate/reuse tests for overlay entries.
- [x] 1.3.10 Add generation bump coverage for all specified events.
- [x] 1.3.11 Add mount removal invalidation tests.
- [x] 1.3.12 Add a review checkpoint proving inode logic can describe future non-disk-backed entries without redesign.

---

## 2. Server-Core and Memfs Overlay — complete

### 2.1 Phase 3: Direct Disk-Backed Server Core

Design references: [Required Internal Shape for v1](./DESIGN.md), [Functional Requirements](./DESIGN.md), [API Design](./DESIGN.md)

- [x] 2.1.1 Implement `FsServerBuilder`.
- [x] 2.1.2 Implement mount registration and lookup by tag.
- [x] 2.1.3 Implement direct host filesystem handling for all 17 `FsOp` variants.
- [x] 2.1.4 Implement read-only mount enforcement.
- [x] 2.1.5 Emit events through broadcast channel.
- [x] 2.1.5a Ensure `FsOpKind` covers every emitted server operation.
- [x] 2.1.6 Call policy hooks consistently.
- [x] 2.1.7 Keep mount/backing representation narrow enough for future non-disk mount types.
- [x] 2.1.8 Introduce internal mount-state abstraction separating tag, overlay, inode, and backing.
- [x] 2.1.9 Confine `std::fs` path resolution to backing-oriented helpers only.
- [x] 2.1.10 Represent each mount tag as an ordered stack with one base-layer slot.
- [x] 2.1.11 Add integration tests against `tempfile` directories for each `FsOp`.
- [x] 2.1.12 Add read-only enforcement tests.
- [x] 2.1.13 Add event emission tests.
- [x] 2.1.13a Add event-emission coverage proving `Setattr` and `Readdir` produce the expected `FsOpKind`.
- [x] 2.1.14 Add multi-tag routing tests.
- [x] 2.1.15 Add dynamic mount add/remove tests.
- [x] 2.1.16 Design guardrail: disk fallback is isolated behind mount/backing logic.
- [x] 2.1.17 Source-level verification: overlay/client modules do not call `std::fs`.
- [x] 2.1.18 Review checkpoint: every mount tag modeled as stack abstraction.

### 2.2 Phase 4: Memfs Overlay Core

Design references: [Ordered Layer-Stack Model](./DESIGN.md), [Product Requirement: Dynamic Runtime Mutation](./DESIGN.md), [In-Memory Layers in an Ordered Mount Stack](./DESIGN.md), [API Design](./DESIGN.md)

- [x] 2.2.1 Implement overlay layers with priority ordering.
- [x] 2.2.1a Implement memfs-layer ordering as an ordered stack above the base layer.
- [x] 2.2.2 Implement `OverlayEntryKind::{Content, Whiteout, SyntheticDir}`.
- [x] 2.2.3 Implement `put_layer`, `remove_layer`, and `layers`.
- [x] 2.2.4 Implement `OverlayAttrs`, `OverlayMutation`, `apply_batch`, `put`, `put_with_attrs`, `whiteout`, `remove`, `get`, `list_layer`, `resolve`, and `list_effective`.
- [x] 2.2.5 Recursively materialize synthetic parent directories.
- [x] 2.2.6 Refcount and prune synthetic directories when children disappear.
- [x] 2.2.7 Implement generic stack resolution for `lookup`, `getattr`, and `read`.
- [x] 2.2.8 Implement generic stack merge for `readdir`.
- [x] 2.2.9 Implement mutation semantics for `Unlink`, `Create`, `Mkdir`, `Rmdir`, `Rename`, and `Setattr`.
- [x] 2.2.9a Implement the v1 symlink/readlink ENOTSUP rule for overlay-managed paths.
- [x] 2.2.10 Implement transparent cross-layer rename behavior from the DESIGN.
- [x] 2.2.11 Implement inherited synthetic ownership defaults.
- [x] 2.2.11a Implement overlay file-handle bookkeeping: synthetic fh, fh→inode mapping, Release, Fsync.
- [x] 2.2.12 Keep overlay node management independent from concrete base-layer details.
- [x] 2.2.13 Ensure overlay entry identity uses tag + mount-relative path.
- [x] 2.2.14 Ensure one named memfs layer can hold entries for many tags.
- [x] 2.2.14a Implement per-tag writer serialization.
- [x] 2.2.14b Implement per-tag atomic memfs snapshot publish.
- [x] 2.2.14c Ensure read-like ops use one stable snapshot per request.
- [x] 2.2.14d Keep batch atomicity scoped to memfs layers only.
- [x] 2.2.14e Use `arc-swap`, `parking_lot`, `bytes` for implementation scaffolding.
- [x] 2.2.14f Do not adopt `vfs` crate or Theseus `memfs`.
- [x] 2.2.15 Layer priority resolution tests.
- [x] 2.2.16 Synthetic parent creation tests.
- [x] 2.2.17 Whiteout suppression tests.
- [x] 2.2.18 `remove_layer` effective-winner change tests.
- [x] 2.2.19 Generation bump tests.
- [x] 2.2.20 Cross-layer rename tests (base→memfs, memfs→base, memfs→memfs).
- [x] 2.2.22 Partial-overlay tests (disk + memfs children).
- [x] 2.2.23 Fully synthetic subtree tests.
- [x] 2.2.24 Whiteout tests against immutable lower entries.
- [x] 2.2.25 Transparent app-workflow tests (create/write/rename/unlink).
- [x] 2.2.25a Symlink/readlink ENOTSUP tests for overlay paths.
- [x] 2.2.26 Shared-tag path coexistence tests.
- [x] 2.2.27 Separate-tag overlay-isolation tests.
- [x] 2.2.28 Shared-layer multi-tag tests.
- [x] 2.2.29 Inherited-ownership tests.
- [x] 2.2.30 Explicit-attr override tests.
- [x] 2.2.31 Future-compatibility review checkpoint.
- [x] 2.2.32 Source-level verification: overlay APIs use mount-relative paths only.
- [x] 2.2.33 Batch atomicity tests.
- [x] 2.2.34 Synthetic-parent atomicity tests.
- [x] 2.2.35a Overlay file-handle lifecycle tests (Open→Read→Write→Fsync→Release→EBADF).
- [x] 2.2.36 Review checkpoint: base layer keeps native fs semantics.

---

## 2.3 Phase 5: Deterministic Cache/TTL Behavior — not started

Design references: [v1 no-caching principle](./DESIGN.md), [FUSE client mount policy](./DESIGN.md)

Primary file targets: `libs/vfs/src/core/server.rs`, `libs/vfs/src/client/fuse.rs`, `libs/vfs/src/core/op.rs`

- [ ] 2.3.1 Ensure disk attrs are re-fetched each time in v1.
- [ ] 2.3.2 Ensure no server-side readdir/data caches exist.
- [ ] 2.3.3 Emit `ttl_secs = 0` consistently where specified.
- [ ] 2.3.4 Encode the v1 mount behavior required for correctness-first mode.
- [ ] 2.3.5 Add mutation-visibility-on-next-operation tests.
- [ ] 2.3.6 Add zero-TTL response assertions.
- [ ] 2.3.7 Add no-stale-view tests across admin-driven mutations in direct mode.

---

## 4. v1 Guest Path

### 4.1 Phase 8: vsock Composite — complete

Design references: [v1: libs/vfs Core Crate + Proof of Concept Examples](./DESIGN.md), [High-Level System Design](./DESIGN.md), [Guest-Side API Boundary](./DESIGN.md), [Guest Agent Code with motlie-vfs](./DESIGN.md)

- [x] 4.1.1 Implement bincode-over-stream request/response handling.
- [x] 4.1.2 Implement `VsockConnectionHandler`.
- [x] 4.1.3 Implement `VsockClientTransport`.
- [x] 4.1.4 Preserve the boundary that VMM handshake remains outside this crate.
- [x] 4.1.5 Add duplex transport tests for handler and guest transport adapter.
- [x] 4.1.6 Add parity tests versus direct server-core behavior.
- [ ] 4.1.7 Add a Cloud Hypervisor-backed smoke test or documented harness procedure.
- [ ] 4.1.8 Add explicit documented steps or scripts for launching a CH guest.

### 4.2 Phase 9: FUSE Client — not started, critical path

Design references: [FR-1: Cross-Platform FUSE Client](./DESIGN.md), [Guest-Side API Boundary](./DESIGN.md), [API Design](./DESIGN.md)

Primary file targets: `libs/vfs/src/client/fuse.rs`, `libs/vfs/src/client/guest.rs`, `libs/vfs/src/client/mod.rs`

- [x] 4.2.1 Map `fuser` callbacks to `FsOp`.
- [x] 4.2.2 Map `FsResult` back to `fuser` reply types.
- [x] 4.2.3 Encode zero-TTL policy.
- [x] 4.2.4 Expose/document the mount option set for v1 correctness-first mode.
- [ ] 4.2.5 Implement build-time FUSE dependency checks for Linux (macOS deferred to v2).
- [x] 4.2.5a Implement `GuestMountSpec` and `GuestMountRunner` in `client/guest.rs`.
- [x] 4.2.5b Ensure `GuestMountRunner` consumes caller-supplied stream/transport connectors.
- [x] 4.2.5c Keep `bins/motlie-vfs-guest.rs` thin: call `GuestMountRunner` rather than reimplementing.
- [ ] 4.2.6 Add callback translation unit tests.
- [ ] 4.2.7 Add mock transport tests.
- [x] 4.2.7a Add unit tests for `GuestMountRunner` using mock transport/connector closures.
- [ ] 4.2.8 Add FUSE integration tests where the environment supports it.
- [x] 4.2.9 macOS FUSE-T is v2 roadmap work, not a v1 requirement.
- [ ] 4.2.10 Add an end-to-end mounted-subtree scenario.

---

## 5. Integration (v1 subset)

### 5.1 Phase 10: Workspace Integration and CH Harness — not started

Design references: [v1: libs/vfs Core Crate + Proof of Concept Examples](./DESIGN.md), [Cloud Hypervisor fast-path / v1 operational setup](./DESIGN.md), [Components and Testing](./DESIGN.md)

- [ ] 5.1.1 Add `motlie-vfs` as a workspace dependency where appropriate.
- [ ] 5.1.2 Ensure feature flags are cleanly consumable.
- [ ] 5.1.3 Validate the design mapping back to `motlie-vmm`.
- [ ] 5.1.4 Update related docs after implementation snapshots land.
- [ ] 5.1.5 Add an implementation-readiness checklist.
- [ ] 5.1.7 Add setup instructions for building `motlie-vfs-guest` and the stacked-root guest image.
- [ ] 5.1.8 Add setup instructions for creating host backing directories.
- [ ] 5.1.9 Add setup instructions for starting the host-side `FsServer` and example harness.
- [ ] 5.1.11 Add explicit CH launch/stop commands or scripts.
- [ ] 5.1.11a Document the guest boundary in harness instructions.
- [ ] 5.1.12 Run workspace `cargo check`.
- [ ] 5.1.13 Run feature-matrix build verification.
- [ ] 5.1.14 Validate the concrete VMM example: boot, mount, inject, verify.
- [ ] 5.1.15 Validate non-overlaid files pass through unchanged.
- [ ] 5.1.18 Validate operator-facing path contract examples.
- [ ] 5.1.19 Validate explicit synthetic ownership.
- [ ] 5.1.20 Validate the whiteout workflow.
- [ ] 5.1.21 Validate the `rm` workflow.
- [ ] 5.1.22 Validate SSH guest access against the v1 test image.
- [ ] 5.1.23 Validate the documented CH launch and stop procedure.
- [ ] 5.1.24 Document the v1 shared `(tag, path)` ownership limitation.
- [ ] 5.1.25 Validate the in-process VMM/REPL hosting pattern.
- [ ] 5.1.26 Validate the Cloud Hypervisor fast-path harness.

---

## Delivery Order

1. ~~1.1 → 1.2 → 1.3~~ (done)
2. ~~2.1 → 2.2~~ (done)
3. ~~4.1~~ (done)
4. **4.2** ← critical path: FUSE client + GuestMountRunner
5. 2.3 — cache/TTL hardening (can parallel with 4.2)
6. 5.1 — CH harness + operational validation

## Excluded from v1

The following PLAN phases are v1.5/v2 and are not in this plan:

- Phase 3.1 (v1.5): Embedded Admin Console + Script / Config Ingestion
- Phase 3.2 (v2): External RPC / gRPC Application Layer
- PLAN tasks 5.1.6 (benchmarks), 5.1.10 (embedded admin procedures), 5.1.16–5.1.17 (latency/throughput baselines)
