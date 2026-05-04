# Apple Vz Vertical-Slice Handoff: Trait-Shape Findings

Addresses [#215 acceptance criterion 7](https://github.com/chungers/motlie/issues/215). This
document captures what the v1.35 and v1.45 Apple Vz vertical slices empirically
proved about the platform-adapter trait shapes that issues
[#169](https://github.com/chungers/motlie/issues/169) (motlie-vnet),
[#171](https://github.com/chungers/motlie/issues/171) (motlie-vfs), and
[#207](https://github.com/chungers/motlie/issues/207) (motlie-vmm/vz native
runner) will need to commit to. It also lists what remains ambiguous so those
issues can resolve their own AC items without re-deriving context.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-04 | @codex-vz | Add v1.5 update: VZ userspace egress is now embedded in the VMM runtime, so the open #169 question narrows to adapter shape and transport, not helper-process ownership |
| 2026-04-26 | @opus47-mac | Initial handoff doc capturing v1.35/v1.45 empirical findings for #169 / #171 / #207 |

## Source of evidence

The findings here come from the merged `feature/vmm-vz` line:

- `libs/vmm/examples/v1.35/` — multi-guest + SSH proxy slice (CH `v1.3` parity)
- `libs/vmm/examples/v1.45/` — auto-provision-via-SSH slice (CH `v1.4` parity)
- `libs/vmm/src/{backend,runtime,orchestrator,artifacts,network}.rs` — backend-owned
  dispatch landed in PR #222
- `libs/vfs/src/core/{inode,op,overlay,server}.rs` — shared VFS core hardening
  landed in PR #222
- `libs/vnet/examples/v1.25/` — Apple Vz egress helper (libslirp userspace bridge)

End-to-end validation of the live shape is in
[PR #222 e2e validation comment](https://github.com/chungers/motlie/pull/222#issuecomment-4322929255).
The current backend split, as of `feature/vmm-vz` head:

```text
HypervisorBacking::AppleVirtualizationShell(VzShellBackend)
FilesystemBacking::MotlieVfs(MotlieVfsBacking)
NetworkBacking::VzUserspaceEgress(VzUserspaceEgressBacking) // v1.5
ControlPlaneBacking::MotlieSshProxy
```

Before v1.5, the VZ slice used `NetworkBacking::HypervisorManaged` and spawned
one `vz_egress_helper` process per guest. v1.5 moves that libslirp role into
the VMM runtime so it is lifecycle-owned like the CH `MotlieVnet` handle. The
external VZ process boundary that remains is `vz-vsock-runner`, parallel to the
CH `cloud-hypervisor` runner boundary.

## #169 (motlie-vnet core/adapter split) — what the slice proved

The Vz egress data path is implemented as a libslirp-backed userspace helper
(`libs/vnet/examples/v1.25/vz_egress_helper.rs`) bridged to the guest NIC via a
UNIX SOCK_DGRAM socketpair owned by `libs/vmm/examples/v1.45/vz-vsock-runner.m`.
The CH egress path uses vhost-user against the Linux-only `motlie-vnet`
crate. They share libslirp as the actual NAT/DNS engine; they differ in
transport.

**Confirmed shape:**

- The reusable core is the libslirp engine plus its stable contract: `(in_frame,
  out_frame)` pairs, plus DNS / DHCP / TCP-stack configuration.
- The platform adapter is the transport: vhost-user-virtio (CH) vs
  UNIX-socketpair-bridged-to-`VZFileHandleNetworkDeviceAttachment` (Vz).
- Backend identity, not network mode, must drive transport selection. PR #222
  proved this — `EgressNetMode::VzUserspace` is rejected by the generic CH
  validator (`validate_network_modes`) and only accepted by
  `validate_vz_network_modes` when `BackendKind::Vz`. Replicate this rule when
  splitting the core.

**Still ambiguous:**

- **Per-guest isolation is selected.** v1.5 keeps one libslirp instance per
  guest, but embeds that instance in the VMM runtime instead of spawning one
  helper process per guest. #169 should focus on the transport/core adapter
  shape and preserve this per-guest isolation unless a later design explicitly
  replaces it.
- **Two-hop bridge cost.** The current path is `VZ NIC fd → socketpair → bridge
  thread → sendto(helper)`. PR #212 finding #7 (still open in #215 AC #4)
  proposes collapsing to per-guest `connect(SOCK_DGRAM)` + direct
  `VZFileHandleNetworkDeviceAttachment`. Whether that collapse should happen
  before or as part of the core/adapter split is a #169 scope call.
- **Naming.** "vnet" vs "egress" vs "userspace egress" appear in different
  surfaces. CONVERGENCE.md picked "Vz userspace egress" for the v1.45
  user-facing text. #169 should land a single canonical term.

## #171 (motlie-vfs core/adapter split) — what the slice proved

Both backends use `FilesystemBacking::MotlieVfs` (the same `FsServer` from
`libs/vfs/src/core/`). They differ only in the transport that carries
`FsOp`/`FsResult` between guest and host:

- CH uses vsock via vhost-vsock.
- Vz bridges through `vz-vsock-runner`'s vsock-forwarding code path
  (`-vsock-forward port:path`), terminating on the same UNIX socket the
  `FsServer` is listening on.

**Confirmed shape:**

- The reusable core is the FUSE-shaped op/result model — `FsOp`, `FsResult`,
  the inode table, the overlay layers, xattr handling, and lock semantics. All
  of these live under `libs/vfs/src/core/`.
- The platform adapter is the transport plus host-side semantic translation
  (e.g., macOS xattr filtering and errno normalization landed in PR #222).
- PR #222 demonstrated that the core can be hardened in CH-visible ways
  (overlay symlinks under synthetic parents, overlay write/rename uid/gid/mode
  preservation) without coupling — both backends got the improvements.

**Still ambiguous:**

- **Where macOS-host adapter logic lives.** PR #222 placed
  `normalize_guest_xattr_errno` and `filter_macos_xattr_list` inside
  `libs/vfs/src/core/server.rs` behind `#[cfg(target_os = "macos")]`. That is
  pragmatic but blurs the core/adapter line. #171 should decide whether host
  errno mapping is a core concern (because the wire contract is "Linux errno
  values for the Linux guest") or a platform adapter concern (because the
  mapping only exists on macOS hosts).
- **Single core, multiple transports.** The current `MotlieVfs` is one server
  type; transport choice is implicit. A clean split likely needs an explicit
  `VfsTransport` trait. #171 should design that surface from the existing
  vsock + UNIX-socket call sites.
- **Acceptance-criterion 6 of #215.** PR #222 deliberately edited
  `libs/vfs/src/core/{inode,op,overlay,server}.rs` (~430+ added lines), which
  is technically inconsistent with #215's "byte-identical CH surface" criterion.
  #171 should either ratify the criterion as "platform-neutral hardening is
  allowed" or carve those changes out into the platform adapter and revert the
  shared edits.

## #207 (motlie-vmm/vz native runner architecture) — what the slice proved

The `vz-vsock-runner.m` Obj-C process is the canonical native-runner shape: it
owns the `VZVirtualMachine`, its dispatch queue, the network attachment,
serial console, and vsock forwards. It is spawned by `launch-vz.sh` which is
in turn invoked by `VzShellBackend::boot()`. It must be codesigned with
`com.apple.security.virtualization`.

**Confirmed shape:**

- Process model: one runner per VM. Multi-guest is multiple runner processes,
  each tied to one `VzShellHandle` in the `BackendHandle` enum. PR #222 added
  `VzShell` next to `ChShell`.
- Lifecycle: `boot` materializes `launch.sh`, executes it (the script exec's
  `launch-vz.sh` which exec's the runner). PID is captured in
  `vz-runner.pid`. Shutdown goes through `shutdown-vz.sh` and now (per this
  PR) honors `stopWithCompletionHandler:` on SIGINT/SIGTERM so destructors
  run and the disk synchronization mode (`VZDiskImageSynchronizationModeFsync`)
  flushes before exit.
- Build / sign: `build-vz-runner.sh` + `sign-vz-runner.sh`. Adhoc-signed-with-
  entitlement is sufficient on macOS 12+; an Apple Development identity is not
  required for the entitlement.

**Still ambiguous:**

- **Final home.** The runner currently exists in three copies:
  `libs/vnet/examples/v1.25/vz-vsock-runner.m`,
  `libs/vmm/examples/v1.35/vz-vsock-runner.m`,
  `libs/vmm/examples/v1.45/vz-vsock-runner.m`.
  Each has diverged slightly. #207 needs to pick the permanent location
  (candidates per #215: `libs/vmm/platforms/apple_vz/` or similar) and a
  consolidation policy. v1.45 is the most up-to-date copy and should be the
  source of truth at the moment of #207's resolution.
- **Implementation language.** The runner is Obj-C. Alternatives include
  Swift (Apple's preferred surface for VZ.framework) and direct Rust via
  `objc2` / `block2` crates calling VZ.framework FFI. Each has tradeoffs for
  packaging, error propagation, and async.
- **Backend variants.** `BackendKind::Vz` currently maps to
  `AppleVirtualizationShell(VzShellBackend)` (shell-out to the runner). Other
  variants are reserved (`AppleVirtualization` placeholder; `ChForkExec`,
  `ChVmmThread` are CH parallels). #207 should describe whether an in-process
  Vz backend is on the roadmap or whether shell-out is final.
- **Shutdown path coupling.** `shutdown-vz.sh` is currently invoked by
  `VzShellBackend::shutdown` and reads pid files written by `launch-vz.sh`.
  After this PR (`@opus47-mac/vmm-vz-215-followup-...`), `kill_stale_runners`
  treats `RUNNER_PID_FILE` as authoritative. #207 should make explicit whether
  the relocated runner keeps a shell-driven shutdown or migrates to direct
  framework calls.

## Convergence test obligations the split must preserve

These are validated today by the v1.35/v1.45 scenarios and must keep passing
across the split:

1. Multi-guest concurrent boot with disjoint subnets (`multiguest-validate.json`).
2. Per-guest VFS isolation: a guest sees only its own host-mounted workspace.
3. Per-guest egress isolation: each guest's default route is its own
   slot-allocated subnet (`10.0.{2+slot}.2`).
4. Unknown-principal SSH first-contact triggers auto-provision; second SSH
   reuses the same VM (`auto-provision-ssh.json`).
5. Tart-free runtime: the post-shutdown `~/.tart/` catalog must remain
   untouched. In v1.5 the only default Vz-side child process is
   `vz-vsock-runner`; VZ egress is VMM/harness-owned runtime state.
6. Convergence contract phases (`CONVERGENCE.md`):
   `image-ready → seed-ready → launched → interactive-ready →
   validation-complete → shutdown-clean`. The `interactive-ready` gate must
   not hide validation work (apt, cargo, package-manager polls).

## What this doc does not decide

This is a handoff, not a decision. Each of #169, #171, #207 should land its
own DESIGN/PLAN that picks among the ambiguities above. The shape findings
here are intended as inputs to those decisions, not substitutes.
