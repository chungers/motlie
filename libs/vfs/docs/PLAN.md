# motlie-vfs Layered Guest Filesystem Composition Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-28 | @codex-pm | Resolve PR #117 round-3 follow-ups: fix stale traceability phase references, add explicit PLAN coverage for v1 symlink/file-handle behavior, and align the plan with the completed event-emission contract |
| 2026-03-28 | @codex-pm | Address PR #117 review feedback: convert DESIGN links to relative paths, fix phase/task numbering, and make the v2 RPC phase specification-oriented instead of implementation-oriented |
| 2026-03-28 | @codex-pm | Align the plan title with the product framing: layered guest filesystem composition rather than transport plumbing |
| 2026-03-28 | @codex-pm | Resolve guest boundary in PLAN: bootstrap/binary delivery stay VMM-owned, `client/guest.rs` owns guest mount orchestration, and the real guest binary must stay a thin wrapper over public guest APIs |
| 2026-03-28 | @codex-pm | Treat the guest-side mounter as a real v1 binary: move it from `examples/` to `bins/motlie-vfs-guest.rs` and make build-before-image assembly explicit in the plan |
| 2026-03-28 | @codex-pm | Resolve v1 module split in PLAN: `client/fuse.rs` owns `FuseClient`, `vsock/` owns transport/handler glue, and `libs/vfs/examples/` contains the host REPL and guest harness binaries |
| 2026-03-28 | @codex-pm | Add v1 memfs concurrency work: batch-first API, per-tag atomic snapshot publish, non-transactional base layer semantics, and implementation scaffolding for helper crates not chosen as foundations |
| 2026-03-28 | @codex-pm | Converge PLAN on the ordered layer-stack model: per-mount stacks in v1 are `N` memfs layers plus one disk base, with shared named layers spanning multiple mount tags via `(layer, tag, path)` keys |
| 2026-03-28 | @codex-pm | Add explicit implementation guardrails and review gates so v1 code cannot accidentally internalize a permanently disk-backed mount model |
| 2026-03-28 | @codex-pm | Add explicit forward-compatibility tasks for v1.5 console/script parity and v2 diskless memfs-tree support so v1 implementation does not block the roadmap |
| 2026-03-28 | @codex-pm | Simplify roadmap tooling choices: `rustyline` for v1 examples, embedded console plus script/config ingestion for v1.5, and v2 microservice API for diskless memfs tree construction |
| 2026-03-28 | @codex-pm | Reorganize PLAN around roadmap slices: v1 core crate + examples, v1.5 embedded admin console + script/config ingestion, v2 external gRPC/RPC app layer |
| 2026-03-28 | @codex-pm | Add explicit v1 operational tasks for guest image build, Cloud Hypervisor launch, host setup, and embedded admin mutation workflows |
| 2026-03-28 | @codex-pm | Prioritize v1 VM delivery on the vsock path and add Cloud Hypervisor as the fastest guest dev/test harness before full VMM integration |
| 2026-03-28 | @codex-pm | Clarify v1 scope: direct Rust/admin loop only inside the crate; v1.5 stays embedded-admin oriented; in-process VMM/REPL hosting verified |
| 2026-03-28 | @codex-pm | Add execution coverage for synthetic uid/gid attrs, mount-relative path contract, and future batch/script ingestion support in embedded admin workflows |
| 2026-03-28 | @codex-pm | Add traceability and verification for shared mounted trees, multi-user path scoping, and v1 ownership limitations |
| 2026-03-28 | @codex-pm | Add explicit DESIGN-to-PLAN traceability and scenario verification for mounted-subtree overlay use cases |
| 2026-03-28 | @codex-pm | Reformat PLAN to comply with AGENTS.md: numbered phases/tasks, checkboxes, and concrete test gates |
| 2026-03-28 | @codex-pm | Initial phased execution plan for motlie-vfs implementation and tracking |

## Purpose

This plan converts [`DESIGN.md`](./DESIGN.md) into an implementation and tracking sequence that can be executed without inventing architecture during delivery.

Roadmap slices:

- `v1`: `libs/vfs` core crate plus proof-of-concept examples under `libs/vfs/examples`
- `v1.5`: embedded admin console + script/config ingestion built on top of the crate
- `v2`: external gRPC / RPC application layer built on top of the crate

Roadmap-specific implementation choices:

- `v1` examples use `rustyline` for the simple REPL
- `v1.5` extends the embedded admin path with:
  - console improvements
  - script/config-driven batch ingestion
  - reloadable file-mapping workflows
- `v2` targets a gRPC/microservice layer that can construct memfs file trees without the
  local disk backing assumptions used in `v1` and `v1.5`

Forward-compatibility requirements:

- `v1` implementation must not hard-code assumptions that prevent a future diskless mount type
- `v1.5` mutation surface must be defined once and exercised identically through both the
  interactive console and script/config ingestion paths

Implementation guardrails for v1:

- mount tag and mount backing must be treated as separate concepts internally
- overlay code must operate on mount-relative paths, not raw host filesystem paths
- `std::fs` access must be concentrated behind a mount-backing boundary
- public semantics should describe “fallback to mount backing,” not assume “fallback to disk”
- verification must include explicit review gates preventing `host_path` assumptions from
  leaking into overlay, inode, client, or future admin-layer semantics
- each mount tag must resolve through an ordered stack:
  - zero or more memfs layers
  - exactly one disk-backed base layer in v1
- one named memfs layer may carry entries for many mount tags, keyed by `(layer, tag, path)`
- memfs mutation is batch-first:
  - batch of 1 is the primitive behind single-file mutation
  - commit is an atomic publish per mount tag
  - non-memfs layers keep native semantics

v1 priorities:

- correctness-first
- deterministic runtime behavior
- file-driven memfs overlay
- zero/near-zero caching semantics as specified in the DESIGN
- small vertical slices with runnable verification

## Status Model

- `[ ]` not started
- `[x]` completed and verified
- blocked work should be annotated inline with `@codex YYYY-MM-DD -- blocked: ...`

## Success Criteria

The v1 effort is complete when all of the following are true:

- `libs/vfs` exists as a workspace crate and builds cleanly behind the intended feature flags
- server-core handles the full `FsOp` set defined in the DESIGN
- memfs overlay semantics match the DESIGN, including `SyntheticDir`, `Whiteout`, generation semantics, and zero-TTL behavior
- vsock works over duplex tests and is ready for `motlie-vmm` integration
- the `v1` guest FUSE path works on Linux for the vsock/Cloud Hypervisor workflow
- proof-of-concept examples under `libs/vfs/examples` can host a simple REPL + file server + CH guest workflow
- the v1 proof-of-concept REPL works through `rustyline`
- memfs batches are atomically published per mount tag and readers do not observe partial batch visibility
- tests concretely verify the functional and non-functional requirements called out in the DESIGN
- the v1 VM guest path is proven over vsock before full `motlie-vmm` orchestration exists

## Traceability to DESIGN

The implementation is not complete unless each DESIGN requirement is covered by both:

- at least one implementation phase/task
- at least one validation or verification step

Product requirement coverage:

- [ ] T.1 Operator-controlled guest filesystem layout is implemented by phases `2.1`, `3.2`, `4.1`, `4.2`, and `5.1`, and verified with mounted-subtree integration scenarios.
- [ ] T.2 Partial overlay of existing guest paths is implemented by phase `2.2` and verified with disk+overlay directory tests inside an existing mounted subtree.
- [ ] T.3 Fully synthetic guest paths are implemented by phase `2.2` and verified with recursive synthetic-parent tests.
- [ ] T.4 Application-specific runtime injection is implemented by phases `2.2` and `3.1`, and verified with runtime layer add/update/remove scenarios.
- [ ] T.5 Shared mounted trees and multi-user paths are implemented by phases `2.1`, `2.2`, and `4.1`, and verified for both shared-tag and separate-tag cases.
- [ ] T.5a Shared named memfs layers across multiple mount tags are implemented by phase `2.2` and verified with identical relative-path injections into distinct tags.
- [ ] T.6 Synthetic uid/gid ownership control is implemented by phases `1.2`, `2.2`, `3.1`, and `4.1`, and verified for both inherited-default and explicit-attr cases.
- [ ] T.7 Dynamic runtime mutation is implemented by phases `2.2` and `3.1`, and verified with next-operation visibility tests.
- [ ] T.7a Atomic memfs mutation visibility is implemented by phase `2.2` and verified with batch-commit visibility tests for lookup, read, and readdir.
- [ ] T.8 Transparency to guest applications is implemented by phases `2.2`, `2.3`, and `4.2`, and verified with rename/unlink/editor-style workflows.
- [ ] T.9 Frontend layering is preserved: `MemOverlay` is core, v1 proof-of-concept examples call it directly, v1.5 extends the embedded admin path, and embedders can host in-process admin loops directly on the Rust API.
- [ ] T.10 v1 VM guest delivery is proven on the vsock path, using Cloud Hypervisor as the fast integration harness before full VMM integration.
- [ ] T.11 v1 operational setup is documented and testable: image build, CH launch, host server setup, and embedded admin mutation workflow.
- [ ] T.12 Roadmap tooling choices are reflected in implementation planning: `rustyline` for v1, embedded console plus script/config ingestion for v1.5, and a diskless memfs-tree microservice direction for v2.
- [ ] T.13 Forward-compatibility is preserved: v1 core implementation leaves room for future non-disk mount backing, and v1.5 console/script parity is explicitly tested.
- [ ] T.14 Implementation guardrails are enforced: v1 code keeps mount/backing separate, isolates `std::fs` behind backing logic, models per-tag ordered stacks, and avoids raw `host_path` assumptions in overlay semantics.
- [ ] T.14a v1 implementation scaffolding uses focused helper crates where useful (`arc-swap`, `parking_lot`, `bytes`, optional `slotmap`) and does not take hard dependencies on rejected foundation crates such as `vfs` or Theseus `memfs`.

Functional requirement coverage:

- [ ] T.15 FR-1 roadmap placement is respected: phase `4.2` proves the Linux guest FUSE path for `v1`, while macOS FUSE-T remains later roadmap work.
- [ ] T.16 FR-2 roadmap placement is respected: transport-agnostic protocol is deferred to phase `3.2` in `v2`.
- [ ] T.17 FR-3 roadmap placement is respected: pluggable wire encoding is deferred to phase `3.2` in `v2`.
- [ ] T.18 FR-4 Tag-based mount routing is implemented by phases `2.1` and `3.2` and verified with multi-tag routing tests.
- [ ] T.19 FR-5 Dynamic mount management is implemented by phases `2.1` and `1.3` and verified with mount add/remove and inode invalidation tests.
- [ ] T.20 FR-6 Event emission is implemented by phase `2.1` and verified with non-blocking event tests.
- [ ] T.21 FR-7 Policy hooks are implemented by phases `1.2` and `2.1`/`2.2` and verified with allow/deny and filtered-`readdir` tests.
- [ ] T.22 FR-8 Read-only and read-write mounts are implemented by phase `2.1` and verified with `EROFS` enforcement tests.
- [ ] T.23 FR-9 roadmap split is respected: `v1` proves direct + vsock, `v1.5` proves embedded admin workflows, and `v2` adds the remote/framed protocol path.
- [ ] T.24 FR-10 In-memory overlay with layered content injection is implemented by phases `2.2` and `3.1` and verified with layer, whiteout, synthetic-dir, runtime mutation, ownership, and embedded admin workflow tests.

Non-functional requirement coverage:

- [ ] T.25 NFR-1 latency has a benchmark or timing harness defined before optimization work begins.
- [ ] T.26 NFR-2 throughput has a benchmark or large-file transfer harness defined before optimization work begins.
- [ ] T.27 NFR-3 minimal dependencies is verified by crate feature/build inspection in phases `1.1` and `4.1`.
- [ ] T.28 NFR-4 library-first is verified by exposing all major functionality through library APIs in phases `1.2` through `4.2`.
- [ ] T.29 NFR-5 testability is verified by direct-mode tests, environment-gated Linux FUSE tests, the Cloud Hypervisor guest harness, and documented operator procedures.

## 1. Foundation

### 1.1 Phase 0: Bootstrap Crate and Workspace

Goal:

- create a compilable `libs/vfs` crate with the module layout and feature flags from the DESIGN

Design references:

- [Architectural Layering and Roadmap](./DESIGN.md)
- [Crate Structure](./DESIGN.md)
- [Feature Flags](./DESIGN.md)

Primary file targets:

- `Cargo.toml`
- `libs/vfs/Cargo.toml`
- `libs/vfs/src/lib.rs`
- `libs/vfs/bins/motlie-vfs-guest.rs`
- `libs/vfs/src/core/mod.rs`
- `libs/vfs/src/client/mod.rs`
- `libs/vfs/src/client/guest.rs`
- `libs/vfs/src/vsock/mod.rs`
- `libs/vfs/examples/`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 1.1.1 Add `libs/vfs` to workspace members.
- [x] 1.1.2 Create `libs/vfs/Cargo.toml` with the feature graph described in the DESIGN.
- [x] 1.1.3 Add initial dependency set for core, vsock, client, and example PoC modules.
- [x] 1.1.4 Create public module skeletons with feature gating only, no behavior yet.
- [x] 1.1.5 Ensure `cargo check -p motlie-vfs` succeeds with default features.
- [x] 1.1.6 Add `rustyline` as the v1 proof-of-concept REPL dependency for `libs/vfs/examples`.
- [x] 1.1.6a Add `libs/vfs/examples/simple_host.rs` and `libs/vfs/examples/README.md` scaffolding so the v1 examples location matches the DESIGN exactly.
- [x] 1.1.6b Add `libs/vfs/src/client/guest.rs` scaffolding so guest-side orchestration has a stable public API distinct from `FuseClient`.
- [x] 1.1.6c Add `libs/vfs/bins/motlie-vfs-guest.rs` scaffolding as the real v1 guest-side mounter binary over the public guest APIs.

Tests / verification:

- [x] 1.1.7 Run `cargo check -p motlie-vfs`.
- [x] 1.1.8 Run a workspace `cargo check` to verify the new crate does not break the workspace.

Exit criteria:

- workspace builds with the new crate present
- feature graph matches the roadmap-aligned crate boundary in the DESIGN

### 1.2 Phase 1: Core Types and Error Surface

Goal:

- establish the stable core API and serialization surface used by every composite

Design references:

- [Functional Requirements](./DESIGN.md)
- [Core Types: FsOp and FsResult](./DESIGN.md)
- [API Design](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/core/op.rs`
- `libs/vfs/src/core/event.rs`
- `libs/vfs/src/core/policy.rs`
- `libs/vfs/src/core/mod.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 1.2.1 Implement `FsOp`, `FsResult`, `FileAttr`, `DirEntry`, `FsStats`, and `SetAttrFields`.
- [x] 1.2.2 Implement `FsEvent` and `FsOpKind` (with `from_op()` conversion).
- [x] 1.2.3 Implement `PolicyFn` and `AllowAll`.
- [x] 1.2.4 Use `anyhow::Result` for operational APIs, matching the workspace convention.
- [x] 1.2.5 Add serde coverage for all wire-visible types (`bytes` crate `serde` feature enabled).
- [x] 1.2.6 Ensure the core `FileAttr`/`SetAttrFields` model carries `uid`, `gid`, and `mode` needed by synthetic overlay entries.

Tests / verification:

- [x] 1.2.7 Add serde round-trip tests for `FsOp` and `FsResult`.
- [x] 1.2.8 Add allow/deny tests for `PolicyFn`.
- [x] 1.2.9 Add serde coverage tests for synthetic attrs carrying `uid`, `gid`, and `mode`.

Exit criteria:

- core wire types are stable enough to support server-core, vsock, and future external applications

### 1.3 Phase 2: InodeTable and Path Model

Goal:

- implement the authoritative per-mount inode namespace described in the DESIGN

Design references:

- [Ordered Layer-Stack Model](./DESIGN.md)
- [Inode Management](./DESIGN.md)
- [Generation Semantics](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/core/inode.rs`
- `libs/vfs/src/core/server.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 1.3.1 Implement per-mount `InodeTable`.
- [x] 1.3.2 Support shared inode namespace across disk entries, `Content`, `SyntheticDir`, and whiteout bookkeeping.
- [x] 1.3.3 Implement inode allocation rules for each entry kind.
- [x] 1.3.4 Implement generation tracking rules from the DESIGN.
- [x] 1.3.5 Implement path-to-inode reverse lookup.
- [x] 1.3.6 Implement stable-within-mount-lifetime behavior.
- [x] 1.3.7 Ensure inode data structures do not require every non-overlay inode to carry a concrete `host_path`.

Tests / verification:

- [x] 1.3.8 Add allocate/lookup/reuse tests for disk entries.
- [x] 1.3.9 Add allocate/reuse tests for overlay entries.
- [x] 1.3.10 Add generation bump coverage for all specified events.
- [x] 1.3.11 Add mount removal invalidation tests.
- [x] 1.3.12 Add a review checkpoint proving inode logic can describe future non-disk-backed entries without redesign.

Exit criteria:

- inode behavior matches the DESIGN and no longer requires architectural decisions during server implementation

## 2. Server-Core and Memfs Overlay

### 2.1 Phase 3: Direct Disk-Backed Server Core

Goal:

- implement `FsServer` for direct host-filesystem operations without overlay enabled

Design references:

- [Required Internal Shape for v1](./DESIGN.md)
- [Functional Requirements](./DESIGN.md)
- [API Design](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/core/server.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 2.1.1 Implement `FsServerBuilder`.
- [x] 2.1.2 Implement mount registration and lookup by tag.
- [x] 2.1.3 Implement direct host filesystem handling for `Lookup`, `Getattr`, `Setattr`, `Readdir`, `Open`, `Read`, `Write`, `Create`, `Mkdir`, `Unlink`, `Rmdir`, `Rename`, `Symlink`, `Readlink`, `Release`, `Fsync`, and `Statfs`.
- [x] 2.1.4 Implement read-only mount enforcement.
- [x] 2.1.5 Emit events through broadcast channel.
- [x] 2.1.5a Ensure `FsOpKind` covers every emitted server operation, including `Setattr` and `Readdir`, so FR-6 remains mechanically enforceable.
- [x] 2.1.6 Call policy hooks consistently.
- [x] 2.1.7 Keep mount/backing representation narrow enough that a future non-disk mount type can be introduced without redesigning `MemOverlay`.
- [x] 2.1.8 Introduce an internal mount-state abstraction that separates tag identity, overlay state, inode state, and fallback backing.
- [x] 2.1.9 Confine `std::fs` path resolution and direct disk operations to backing-oriented helpers or modules, not generic overlay/server entry points.
- [x] 2.1.10 Represent each mount tag internally as an ordered stack with one base-layer slot and zero-or-more memfs-layer views above it.

Tests / verification:

- [x] 2.1.11 Add integration tests against `tempfile` directories for each `FsOp`.
- [x] 2.1.12 Add read-only enforcement tests.
- [x] 2.1.13 Add event emission tests.
- [x] 2.1.13a Add event-emission coverage proving `Setattr` and `Readdir` produce the expected `FsOpKind`.
- [x] 2.1.14 Add multi-tag routing tests covering independent mounted subtrees.
- [x] 2.1.15 Add dynamic mount add/remove tests covering tag registration and invalidation behavior.
- [x] 2.1.16 Add a design guardrail review checkpoint asserting disk fallback is isolated behind mount/backing logic, not spread through overlay logic.
- [x] 2.1.17 Add a source-level verification step (`rg`/review checklist) proving overlay/client modules do not directly call `std::fs` or concatenate raw host-root paths.
- [x] 2.1.18 Add a review checkpoint proving every mount tag is modeled as a stack abstraction rather than “overlay + ad hoc disk fallback.”

Exit criteria:

- server-core works in direct mode for disk-only mounts

### 2.2 Phase 4: Memfs Overlay Core

Goal:

- implement the file-driven memfs overlay exactly as specified

Design references:

- [Ordered Layer-Stack Model](./DESIGN.md)
- [Product Requirement: Dynamic Runtime Mutation](./DESIGN.md)
- [In-Memory Layers in an Ordered Mount Stack](./DESIGN.md)
- [API Design](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/core/overlay.rs`
- `libs/vfs/src/core/server.rs`
- `libs/vfs/src/core/inode.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 2.2.1 Implement overlay layers with priority ordering.
- [x] 2.2.1a Implement memfs-layer ordering as an ordered stack above the base layer for each mount tag.
- [x] 2.2.2 Implement `OverlayEntryKind::{Content, Whiteout, SyntheticDir}`.
- [x] 2.2.3 Implement `put_layer`, `remove_layer`, and `layers`.
- [x] 2.2.4 Implement `OverlayAttrs`, `OverlayMutation`, `apply_batch`, `put`, `put_with_attrs`, `whiteout`, `remove`, `get`, `list_layer`, `resolve`, and `list_effective`.
- [x] 2.2.5 Recursively materialize synthetic parent directories.
- [x] 2.2.6 Refcount and prune synthetic directories when children disappear.
- [x] 2.2.7 Implement generic stack resolution for `lookup`, `getattr`, and `read`: top-down through memfs layers, then base layer.
- [x] 2.2.8 Implement generic stack merge for `readdir`: base-layer entries merged upward through memfs layers with whiteout removal semantics.
- [x] 2.2.9 Implement mutation semantics for `Unlink`, `Create`, `Mkdir`, `Rmdir`, `Rename`, and `Setattr`.
- [x] 2.2.9a Implement the v1 symlink/readlink rule explicitly: symlinks remain base-layer-only, and overlay-managed or synthetic-parent symlink operations return `ENOTSUP`.
- [x] 2.2.10 Implement transparent cross-layer rename behavior from the DESIGN.
- [x] 2.2.11 Implement inherited synthetic ownership defaults from nearest existing parent or mount root.
- [x] 2.2.11a Implement overlay file-handle bookkeeping for overlay-backed opens: synthetic `fh` allocation, `fh -> inode` mapping, `Release`, `Fsync`, and latest-effective-content read behavior.
- [x] 2.2.12 Keep overlay node management independent from concrete base-layer implementation details where possible.
- [x] 2.2.13 Ensure overlay entry identity, synthetic parent creation, whiteouts, and listings remain defined in terms of tag + mount-relative path rather than disk-path existence.
- [x] 2.2.14 Ensure one named memfs layer can hold entries for many tags without path collision.
- [x] 2.2.14a Implement per-tag writer serialization for memfs mutation.
- [x] 2.2.14b Implement per-tag atomic memfs snapshot publish so one committed batch becomes visible all at once.
- [x] 2.2.14c Ensure read-like ops (`lookup`, `getattr`, `read`, `readdir`) use one stable memfs snapshot for the duration of each request.
- [x] 2.2.14d Keep batch atomicity scoped to memfs layers only; do not add fake transactional semantics to the base disk layer.
- [x] 2.2.14e Use focused helper crates where appropriate for v1 implementation scaffolding:
  - `arc-swap` for published snapshot pointers
  - `parking_lot` for per-tag writer serialization
  - `bytes` for payload storage
  - optional `slotmap` if stable synthetic node storage benefits from it
- [x] 2.2.14f Do not adopt the `vfs` crate or Theseus `memfs` as foundational runtime dependencies.

Tests / verification:

- [x] 2.2.15 Add layer priority resolution tests.
- [x] 2.2.16 Add synthetic parent creation tests.
- [x] 2.2.17 Add whiteout suppression tests.
- [x] 2.2.18 Add `remove_layer` effective-winner change tests.
- [x] 2.2.19 Add generation bump tests for required mutation cases.
- [x] 2.2.20 Add cross-layer rename tests for base -> memfs, memfs -> base, and memfs -> memfs.
- [ ] 2.2.21 Add policy-filtered `readdir` tests.
<!-- @claude-dev 2026-03-29 -- policy-filtered readdir is exercised through existing policy tests; dedicated overlay+policy readdir test deferred to Phase 2.3 which covers policy/TTL behavior. -->
- [x] 2.2.22 Add partial-overlay tests for an existing mounted subtree such as `/home/alice/.ssh` where some files come from the base layer and some from memfs.
- [x] 2.2.23 Add fully synthetic subtree tests for paths such as `/home/alice/.claude/skills` created entirely by file injection.
- [x] 2.2.24 Add whiteout tests showing a lower immutable file remains hidden without physical deletion.
- [x] 2.2.25 Add transparent app-workflow tests covering create/write/rename/unlink sequences typical of editors and CLI tools.
- [x] 2.2.25a Add symlink/readlink tests proving overlay-managed paths return `ENOTSUP` in v1 while base-layer symlink behavior remains available.
- [x] 2.2.26 Add shared-tag tests proving `/alice/...` and `/bob/...` coexist in one mounted tree without path collision.
- [x] 2.2.27 Add separate-tag tests proving two guest mounts remain overlay-isolated even when one `MemOverlay` instance serves both tags.
- [x] 2.2.28 Add shared-layer multi-tag tests proving one named layer can inject `/CLAUDE.md` into distinct tags with independent effective placement.
- [x] 2.2.29 Add inherited-ownership tests proving synthetic files/dirs pick up `uid`/`gid` from nearest parent or mount root.
- [x] 2.2.30 Add explicit-attr tests proving `put_with_attrs` overrides default synthetic ownership.
- [x] 2.2.31 Add future-compatibility tests or review checkpoints ensuring stack semantics do not assume every mount has local disk backing.
- [x] 2.2.32 Add a source-level verification step proving overlay APIs and tests remain expressed in mount-relative paths, not raw host-disk paths.
- [x] 2.2.33 Add batch atomicity tests proving a batch of injected files becomes visible all at once for `lookup`, `read`, and `readdir`.
- [x] 2.2.34 Add synthetic-parent atomicity tests proving readers cannot observe partially materialized parent hierarchies.
- [ ] 2.2.35 Add concurrent reader/writer tests proving one filesystem request uses one stable memfs snapshot.
<!-- @claude-dev 2026-03-29 -- concurrent snapshot isolation is architecturally enforced by arc-swap; dedicated multi-threaded stress test deferred to integration phase. -->
- [x] 2.2.35a Add overlay file-handle lifecycle tests covering synthetic `fh` allocation, `Release`, `Fsync`, and reads after content replacement.
- [x] 2.2.36 Add a review checkpoint confirming the base layer keeps native filesystem semantics and is not treated as transactional by `motlie-vfs`.

Exit criteria:

- overlay semantics match the final reviewed DESIGN
- memfs batch visibility and snapshot atomicity match the final reviewed DESIGN
- direct mode with overlay enabled is functionally complete

### 2.3 Phase 5: Deterministic Cache/TTL Behavior

Goal:

- encode the v1 no-cache/correctness-first rules in code and tests

Design references:

- [v1 no-caching principle](./DESIGN.md)
- [FUSE client mount policy](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/core/server.rs`
- `libs/vfs/src/client/fuse.rs`
- `libs/vfs/src/core/op.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 2.3.1 Ensure disk attrs are re-fetched each time in v1.
- [x] 2.3.2 Ensure no server-side readdir/data caches exist.
- [x] 2.3.3 Emit `ttl_secs = 0` consistently where specified.
- [x] 2.3.4 Encode the v1 mount behavior required for correctness-first mode.

Tests / verification:

- [x] 2.3.5 Add mutation-visibility-on-next-operation tests.
- [x] 2.3.6 Add zero-TTL response assertions.
- [x] 2.3.7 Add no-stale-view tests across admin-driven mutations in direct mode.

Exit criteria:

- deterministic v1 behavior is enforced, not merely described

## 3. Roadmap Extensions

### 3.1 v1.5 Phase 6: Embedded Admin Console + Script / Config Ingestion

Goal:

- strengthen the local operator/admin workflow without adding a remote admin interface

Design references:

- [v1.5: Embedded Admin Console + Script / Config Ingestion](./DESIGN.md)
- [Overlay Control: Frontend Architecture](./DESIGN.md)

Primary file targets:

- `libs/vfs/examples/simple_host.rs`
- `libs/vfs/examples/README.md`
- docs under `libs/vfs/docs/` that describe the protocol and usage contract

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [ ] 3.1.1 Define the embedded admin command surface for local operator use.
- [ ] 3.1.2 Support local admin commands for `layer`, `rmlayer`, `layers`, `put`, `putattr`, `whiteout`, `rm`, `get`, `ls`, and `lslayer`.
- [ ] 3.1.3 Ensure `ls` / `lslayer` expose `kind`, `inode`, `uid`, `gid`, and `mode`.
- [ ] 3.1.4 Dispatch embedded admin commands directly to `MemOverlay`.
- [ ] 3.1.5 Define a script/config format for batch injection and reload.
- [ ] 3.1.6 Implement script/config-driven batch ingestion through `apply_batch`.
- [ ] 3.1.7 Return stable, parseable errors in both the console flow and the script/config flow.

Tests / verification:

- [ ] 3.1.8 Add command parsing/dispatch tests for the embedded admin console.
- [ ] 3.1.9 Add script/config ingestion tests for batch load and reload.
- [ ] 3.1.10 Add binary payload handling tests where content is sourced from files or inline script/config input.
- [ ] 3.1.11 Add runtime mutation tests for `put`, `putattr`, `whiteout`, `rm`, and `rmlayer` against a live mounted-subtree scenario.
- [ ] 3.1.12 Add parity tests so the embedded console path and the script/config ingestion path expose the same mutation semantics.

Exit criteria:

- embedded admin workflows work end-to-end against the core crate API without introducing a remote admin layer

### 3.2 v2 Phase 7: External RPC / gRPC Application Layer

Goal:

- define the future external remote-control layer above `libs/vfs`

Design references:

- [v2: External gRPC / RPC Application Layer](./DESIGN.md)
- [Roadmap Gaps and Forward-Compatibility Requirements](./DESIGN.md)

Primary file targets:

- outside `libs/vfs`; application-owned sources and transport bindings
- docs under `libs/vfs/docs/` that describe the API boundary needed by that layer

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [ ] 3.2.1 Define `Frame` and request/response wrapping.
- [ ] 3.2.2 Define the `Codec` abstraction.
- [ ] 3.2.3 Specify the default bincode codec shape.
- [ ] 3.2.4 Define length-prefixed frame I/O semantics.
- [ ] 3.2.5 Define client hello/tag binding semantics.
- [ ] 3.2.6 Specify `RpcServer` responsibilities and boundaries.
- [ ] 3.2.7 Specify `RpcClient` responsibilities and boundaries.
- [ ] 3.2.8 Define request/response correlation requirements.
- [ ] 3.2.9 Define the v2 microservice API for constructing memfs file trees without relying on local disk-backed trees.

Tests / verification:

- [ ] 3.2.10 Define codec round-trip validation requirements.
- [ ] 3.2.11 Define max frame size enforcement tests.
- [ ] 3.2.12 Define duplex integration validation requirements.
- [ ] 3.2.13 Define direct comparison tests against server-core behavior.
- [ ] 3.2.14 Define multi-tag handshake/routing validation requirements.
- [ ] 3.2.15 Define a transport-generic proof point showing the RPC layer only requires `AsyncRead + AsyncWrite`.
- [ ] 3.2.16 Add design-validation examples for diskless memfs tree construction semantics in the v2 API.

Exit criteria:

- future RPC / gRPC layer is specified well enough to build without reshaping the core crate

## 4. v1 Guest Path

### 4.1 Phase 8: vsock Composite

Goal:

- implement the minimal VM-oriented path that bypasses RPC framing overhead

Design references:

- [v1: libs/vfs Core Crate + Proof of Concept Examples](./DESIGN.md)
- [High-Level System Design](./DESIGN.md)
- [Guest-Side API Boundary](./DESIGN.md)
- [Guest Agent Code with motlie-vfs](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/vsock/handler.rs`
- `libs/vfs/src/vsock/client.rs`
- `libs/vfs/src/vsock/mod.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 4.1.1 Implement bincode-over-stream request/response handling.
- [x] 4.1.2 Implement `VsockConnectionHandler`.
- [x] 4.1.3 Implement `VsockClientTransport`.
- [x] 4.1.4 Preserve the boundary that VMM handshake remains outside this crate.

Tests / verification:

- [x] 4.1.5 Add duplex transport tests for handler and guest transport adapter.
- [x] 4.1.6 Add parity tests versus direct server-core behavior.
- [x] 4.1.7 Add a Cloud Hypervisor-backed smoke test or documented harness procedure for guest boot + vsock mount + overlay visibility.
- [x] 4.1.8 Add explicit documented steps or scripts for launching a CH guest with vsock and the stacked-root test image.

Exit criteria:

- host/guest integration boundary is clear and testable
- the vsock path is strong enough to support the first VM-backed product without requiring the full VMM first

### 4.2 Phase 9: FUSE Client

Goal:

- implement `FuseClient` as the guest-side `fuser::Filesystem` bridge with the v1 mount policy

Design references:

- [FR-1: Cross-Platform FUSE Client](./DESIGN.md)
- [Guest-Side API Boundary](./DESIGN.md)
- [API Design](./DESIGN.md)

Primary file targets:

- `libs/vfs/src/client/fuse.rs`
- `libs/vfs/src/client/guest.rs`
- `libs/vfs/src/client/mod.rs`
- `libs/vfs/build.rs`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [x] 4.2.1 Map `fuser` callbacks to `FsOp`.
- [x] 4.2.2 Map `FsResult` back to `fuser` reply types.
- [x] 4.2.3 Encode zero-TTL policy.
- [x] 4.2.4 Expose/document the mount option set for v1 correctness-first mode.
- [ ] 4.2.5 Implement build-time FUSE dependency checks for Linux (macOS deferred to v2).
- [x] 4.2.5a Implement `GuestMountSpec` and `GuestMountRunner` in `client/guest.rs` as the public guest orchestration layer above `FuseClient`.
- [x] 4.2.5b Ensure `GuestMountRunner` consumes caller-supplied stream/transport connectors so VMM handshake/bootstrap logic stays outside this crate.
- [x] 4.2.5c Keep `bins/motlie-vfs-guest.rs` thin: it may parse config and obtain streams, but it must call `GuestMountRunner` rather than reimplementing mount orchestration.

Tests / verification:

- [ ] 4.2.6 Add callback translation unit tests.
- [ ] 4.2.7 Add mock transport tests.
- [ ] 4.2.7a Add unit tests for `GuestMountRunner` using mock transport/connector closures.
- [ ] 4.2.8 Add FUSE integration tests where the environment supports it.
- [x] 4.2.9 macOS FUSE-T is v2 roadmap work, not a v1 requirement.
- [ ] 4.2.10 Add an end-to-end mounted-subtree scenario showing guest-visible behavior for partial overlay, synthetic dirs, and whiteouts.

Clarification:

- `client/fuse.rs` owns the guest-side `fuser::Filesystem` implementation
- `client/guest.rs` owns guest-side mount orchestration over public APIs
- `bins/motlie-vfs-guest.rs` is the real v1 guest-side mounter binary, not an example harness
- `vsock/` owns transport and handler glue only

Exit criteria:

- Linux mounting works in practice
- the `v1` Linux guest FUSE path is proven; broader cross-platform client work remains roadmap material

## 5. Integration

### 5.1 Phase 10: Workspace Integration and motlie-vmm Alignment

Goal:

- make the crate consumable by the workspace and ready for downstream integration

Design references:

- [v1: libs/vfs Core Crate + Proof of Concept Examples](./DESIGN.md)
- [Cloud Hypervisor fast-path / v1 operational setup](./DESIGN.md)
- [Components and Testing](./DESIGN.md)

Primary file targets:

- `Cargo.toml`
- `libs/vfs/Cargo.toml`
- related docs under `libs/vfs/docs/`
- guest-harness scripts/docs under a repo-owned path such as `tools/`, `xtask/`, or `scripts/`

Tasks:

Per-task reference rule:

- every task in this phase traces to one or more of the DESIGN references listed above unless an individual task states a narrower reference explicitly

- [ ] 5.1.1 Add `motlie-vfs` as a workspace dependency where appropriate.
- [ ] 5.1.2 Ensure feature flags are cleanly consumable for direct mode, vsock host/guest paths, and future v2 expansion.
- [ ] 5.1.3 Validate the design mapping back to `motlie-vmm`.
- [ ] 5.1.4 Update related docs after implementation snapshots land.
- [ ] 5.1.5 Add an implementation-readiness checklist that maps completed phases back to the DESIGN traceability items.
- [ ] 5.1.6 Add a small benchmark or measurement harness for metadata latency and large-file throughput.
- [ ] 5.1.7 Add explicit setup instructions or scripts for building `motlie-vfs-guest` first, then building the stacked-root guest image used in v1 guest tests.
- [ ] 5.1.8 Add explicit setup instructions or scripts for creating host backing directories and sample disk-backed files for overlay tests.
- [ ] 5.1.9 Add explicit setup instructions or scripts for starting the host-side `FsServer`, `MemOverlay`, and the v1 in-process REPL/example harness.
- [ ] 5.1.10 Add explicit embedded admin console or script/config procedures for `put`, `putattr`, `whiteout`, `rm`, `rmlayer`, `ls`, and `lslayer` in the guest-harness workflow.
- [ ] 5.1.11 Add explicit CH launch/stop commands or scripts, including vsock and block-device wiring for the stacked-root guest image.
- [ ] 5.1.11a Document the guest boundary explicitly in harness instructions: bootstrap/binary delivery remain VMM-owned, while `bins/motlie-vfs-guest.rs` only exercises the public guest APIs from `client/guest.rs`.

Suggested setup snippets to include in this phase:

```bash
# Host backing directories for mounted subtrees
mkdir -p /tmp/motlie-vfs/home-alice/projects
mkdir -p /tmp/motlie-vfs/scratch-alice
printf 'hello\n' > /tmp/motlie-vfs/home-alice/projects/README.md

# Build the v1 host and guest binaries before guest image assembly
cargo build -p motlie-vfs --example simple_host --bin motlie-vfs-guest

# Start the host example / REPL
cargo run -p motlie-vfs --example simple_host
```

```bash
# Example Cloud Hypervisor launch shape for the v1 harness
cloud-hypervisor \
  --kernel ./artifacts/vmlinux \
  --disk path=./artifacts/root.squashfs,readonly=on \
  --disk path=./artifacts/overlay.ext4 \
  --vsock cid=42,socket=/tmp/motlie-vfs-vsock.sock \
  --console tty
```

Concrete end-to-end walkthrough to include in this phase:

```bash
# 1. Build the host REPL and guest mounter binary on the host.
cargo build -p motlie-vfs --example simple_host --bin motlie-vfs-guest

# 2. Prepare host-backed data for the guest /home/alice subtree.
mkdir -p /tmp/motlie-vfs/home/alice
mkdir -p /tmp/motlie-vfs/home/alice/projects
printf 'from-host\n' > /tmp/motlie-vfs/home/alice/projects/README.md

# 3. Generate a test ssh keypair for remote login validation.
ssh-keygen -t ed25519 -N '' -f /tmp/motlie-vfs/id_alice_test

# 4. Build a minimal guest image that includes:
#    - stacked squashfs + ext4 root
#    - sshd enabled
#    - user alice with uid=1000 gid=1000 and home /home/alice
#    - motlie-vfs-guest installed in the guest image
#    - a config file such as /etc/motlie-vfs/mounts.yaml containing:
#
#    mounts:
#      - tag: alice-home
#        guest_path: /home/alice
#        read_only: false
#
#    The image-build script for this phase should make these exact artifacts reproducible.

# 5. Start the host-side v1 REPL / server with tag alice-home bound to the host path.
cargo run -p motlie-vfs --example simple_host
```

```text
# 6. In the host REPL, register a mount equivalent to:
#    tag = alice-home
#    host backing path = /tmp/motlie-vfs/home/alice
#
#    The final REPL/config syntax can vary, but the procedure must support this exact case.
```

```bash
# 7. Boot the guest with Cloud Hypervisor.
#    Requirements for this test:
#    - the guest can reach the host via vsock for motlie-vfs traffic
#    - the guest exposes ssh on a reachable port for validation
#    - example shape: forward host tcp/2222 -> guest tcp/22 using the chosen CH networking helper
cloud-hypervisor \
  --kernel ./artifacts/vmlinux \
  --disk path=./artifacts/root.squashfs,readonly=on \
  --disk path=./artifacts/overlay.ext4 \
  --vsock cid=42,socket=/tmp/motlie-vfs-vsock.sock \
  --net "tap=tap0,mac=02:fc:00:00:00:01" \
  --console tty
```

```text
# 8. After boot, the guest should:
#    - start sshd
#    - run motlie-vfs-guest
#    - mount tag alice-home at /home/alice
#
#    Verify on the guest console (or over serial) that:
#    - mount shows /home/alice as a FUSE mount
#    - /home/alice/projects/README.md is visible and reads "from-host"
```

```text
# 9. In the host REPL, inject alice's authorized_keys file into the mounted subtree.
#    Use uid=1000 gid=1000 mode=0600 for the file, and create parents as needed.
putattr credentials alice-home /.ssh/authorized_keys 1000 1000 0600 <contents of /tmp/motlie-vfs/id_alice_test.pub>
ls alice-home /.ssh
```

```bash
# 10. Validate guest-visible ownership and modes, then connect remotely.
#     The guest-side validation should confirm:
#     - /home/alice/.ssh exists
#     - /home/alice/.ssh/authorized_keys exists
#     - ownership is alice:alice (1000:1000)
#     - file mode is 0600
#
#     Then verify remote ssh from the host:
ssh -i /tmp/motlie-vfs/id_alice_test -p 2222 alice@127.0.0.1
```

```text
# 11. Optional follow-up checks in the same scenario:
#     - inject additional files under /home/alice/.ssh (config, known_hosts)
#     - whiteout a lower-layer file under /home/alice
#     - remove an injected file with rm and confirm it disappears without reboot/remount
```

```text
# Example in-process REPL flow to verify overlay behavior
put credentials home /.ssh/id_ed25519 <bytes>
put credentials home /.ssh/config <bytes>
ls home /.ssh
whiteout credentials home /.ssh/old_config
rm credentials home /.ssh/config
```

Tests / verification:

- [ ] 5.1.12 Run workspace `cargo check`.
- [ ] 5.1.13 Run feature-matrix build verification.
- [ ] 5.1.14 Validate the concrete VMM example: boot static squashfs+ext4 root, mount `/home/alice` through `motlie-vfs`, then inject `/home/alice/.ssh/...` and `/home/alice/.claude/skills/...` dynamically.
- [ ] 5.1.15 Validate that non-overlaid files inside the mounted subtree continue to pass through unchanged.
- [ ] 5.1.16 Record baseline latency measurements for small metadata operations over Unix sockets.
- [ ] 5.1.17 Record baseline throughput measurements for large sequential reads/writes over Unix sockets.
- [ ] 5.1.18 Validate operator-facing path contract examples for both `home + /alice/...` and `alice-home + /.claude/...` forms.
- [ ] 5.1.19 Validate explicit synthetic ownership by injecting entries for provisioned guest users and checking guest-visible `uid`/`gid`.
- [ ] 5.1.20 Validate the `whiteout` / tombstone workflow hides lower disk files from the guest.
- [ ] 5.1.21 Validate the `rm` workflow removes injected synthetic files from the guest view.
- [ ] 5.1.22 Validate SSH guest access against the v1 test image where applicable.
- [ ] 5.1.23 Validate the documented CH launch and stop procedure from a clean host environment.
- [ ] 5.1.24 Document the remaining v1 limitation that a shared `(tag, path)` entry cannot present different uid/gid ownership to different guests simultaneously.
- [ ] 5.1.25 Validate the in-process VMM/REPL hosting pattern: a Tokio task can call `server.overlay()` directly while filesystem serving remains active.
- [ ] 5.1.26 Validate the Cloud Hypervisor fast-path harness as the pre-VMM guest development environment for the vsock flow.

Exit criteria:

- crate is ready for downstream adoption without reshaping public APIs
- all traceability items in this plan can be marked complete without ad hoc interpretation

## 6. Dependency Graph

- [ ] 6.1 `1.1` before everything
- [ ] 6.2 `1.2` before `2.1`, `3.2`, `4.1`, and `4.2`
- [ ] 6.3 `1.3` before `2.1`, `2.2`, and `2.3`
- [ ] 6.4 `2.1` before `2.2`, `3.2`, and `4.1`
- [ ] 6.5 `2.2` before `2.3`, `3.1`, and `4.2`
- [ ] 6.6 `2.3` before final client/integration signoff
- [ ] 6.7 `3.2` and `4.1` may proceed in parallel after `2.1`
- [ ] 6.8 `4.2` depends on `2.3` and either `3.2` or `4.1` for real transport-backed tests
- [ ] 6.9 `5.1` closes out after all functional phases are stable

## 7. Suggested Delivery Order

- [ ] 7.1 `1.1`
- [ ] 7.2 `1.2`
- [ ] 7.3 `1.3`
- [ ] 7.4 `2.1`
- [ ] 7.5 `2.2`
- [ ] 7.6 `2.3`
- [ ] 7.7 `3.1`
- [ ] 7.8 `4.1`
- [ ] 7.9 `4.2` on vsock path first
- [ ] 7.10 `5.1`
- [ ] 7.11 `3.2` after the vsock-backed VM path is stable

Rationale:

- direct-mode server-core plus overlay settles semantics before adding transport complexity
- vsock is the v1 VM delivery path and should be proven before broader transport work
- Cloud Hypervisor provides the fastest guest harness before a full VMM exists
- remote admin remains valuable, but it is not on the critical path for the first VM-backed product

## 8. Risks to Track

- [ ] 8.1 `direct_io` and mount behavior may differ across Linux and macOS/FUSE-T (macOS deferred to v2)
- [ ] 8.2 `fuser` behavior and kernel/FUSE caching semantics need validation against the v1 deterministic model
- [ ] 8.3 cross-boundary rename behavior may expose platform-specific corner cases
- [ ] 8.4 zero-TTL behavior may be correct but slower than expected; do not optimize before parity tests pass

## 9. Out of Scope for v1

- [ ] 9.1 aggressive performance caching
- [ ] 9.2 explicit FUSE invalidation/notification machinery
- [ ] 9.3 alternate codecs beyond bincode
- [ ] 9.4 Windows support
- [ ] 9.5 generalized distributed filesystem features
- [ ] 9.6 library-provided HTTP admin frontend
- [ ] 9.7 library-provided gRPC admin frontend
- [ ] 9.8 full VMM orchestration as a prerequisite for proving the first guest-vsock path

## 10. Implementation Scaffolding Notes

These are implementation recommendations for v1. They guide the implementer, but they are not
public API guarantees by themselves.

- [ ] 10.1 Prefer a per-tag published memfs snapshot model over fine-grained mutable tree locking.
- [ ] 10.2 Prefer `arc-swap` for atomic publication of per-tag memfs snapshots.
- [ ] 10.3 Prefer `parking_lot` for short writer-side critical sections and per-tag writer serialization.
- [ ] 10.4 Use `bytes::Bytes` for file payload storage and sharing.
- [ ] 10.5 Consider `slotmap` only if synthetic node storage benefits from stable internal keys.
- [ ] 10.6 Do not make `im` or other persistent-collection crates mandatory up front; add them only if snapshot copy cost becomes a measured problem.
- [ ] 10.7 Do not take a foundational dependency on the `vfs` crate.
- [ ] 10.8 Do not take a dependency on the Theseus OS `memfs` project; it is unrelated to this design.

## 11. Tracking Update Format

Recommended per-phase update format:

```text
Phase 2.2: in_progress
Files: core/overlay.rs, core/server.rs, core/inode.rs
Tests added: 12
Tests passed: 10
Open issues: whiteout rename edge case
Unlocks: 2.3, 3.1, 4.2
```
