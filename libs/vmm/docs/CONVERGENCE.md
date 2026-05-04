# motlie-vmm Guest Convergence Contract

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-02 | @codex-vz | Add the v1.5 ownership decision: VMM owns the common CH/VZ guest image, seed schema, guest binary packaging, and guest runtime home; VFS/VNET remain reusable subsystem libraries |
| 2026-04-26 | @codex-vz | Make the v1.45 Vz image hardening caveat explicit: apt-daily masking and ForceIPv4 are current Vz slice assumptions, not a converged CH/Vz image contract |
| 2026-04-25 | @codex-vz | Treat prebuilt host Vz runner/egress helper artifacts as launch prerequisites by default; first-contact startup must not hide host cargo builds |
| 2026-04-25 | @codex-vz | Tighten v1.45 Vz first-contact enforcement: service units, SSH CA config, agent-state scripts, CLIs, and packages are base-image contract; runtime only stages dynamic mounts/CA/principal/identity and restarts VFS |
| 2026-04-25 | @codex-vz | Remove v1.45 Vz first-contact dead ends inherited from standalone smoke scripts: no runtime guest builds, no runtime npm repair, and no default per-guest seed DMG creation |
| 2026-04-25 | @codex-vz | Add the cross-backend guest boot/provisioning convergence contract and make the v1.45 Vz auto-provisioning long-pole plan findable from VMM, Vz, and harness docs |

## Purpose

This is the durable contract for converging Cloud Hypervisor and Apple
Virtualization.framework guest behavior across `libs/vmm`, `libs/vfs`, and
`libs/vnet`.

The important requirement is not just that both backends eventually pass the
same validations. They must converge on the same boot and provisioning shape so
multi-guest auto-provisioning has predictable latency, state ownership, and
failure modes across platforms.

## Contract

Every backend must make these phases explicit:

1. image-ready
2. seed-ready
3. launched
4. interactive-ready
5. validation-complete
6. shutdown-clean

`interactive-ready` is the gate used by first-contact SSH auto-provisioning.
It means:

- the guest is booted
- the requested login principal can authenticate through the harness SSH proxy
- required host-backed mounts are attached and readable
- guest-side agent-state setup needed for interactive sessions has completed

`interactive-ready` must not mean:

- package installation completed during first SSH
- Rust crates were compiled in the guest during first SSH
- outbound egress certification completed
- package-manager quiescence was polled
- optional benchmark or certification probes completed

`validation-complete` is an explicit harness/scenario certification step. It
may perform DNS, HTTPS, VFS sentinel, CLI, package-manager, benchmark, or other
slow probes. It must not be hidden in the first SSH path.

## Image And Seed Ownership

Long-pole setup belongs in image building or seed generation, not in
first-contact SSH:

- OS packages and toolchains
- `/usr/local/bin/motlie-vfs-guest`
- baseline coding-agent CLIs and their executable path wiring
- systemd units that are backend-neutral
- stable SSH CA trust configuration where cloud-init/base-image can own it
- stable agent-state bootstrap scripts
- mount declarations and per-guest identity data

Per-guest runtime setup may still happen after launch when it is genuinely
dynamic:

- principal, username, uid, gid, and hostname assignment
- per-guest CA principal file if it is not yet seed/cloud-init owned
- per-guest mount YAML delivered by seed/cloud-init, or by a documented
  transitional inline payload while Vz seed convergence is incomplete
- backend-specific service restarts needed to bind runtime sockets

This split is the convergence requirement. CH and Vz can have different
mechanisms, but they should not hide different long poles in different phases
unless the underlying platform forces that difference and the docs explain why.

## v1.45 Vz Enforcement

The v1.45 Vz launcher now enforces the first slice of this contract:

- `build-guest.sh` masks `apt-daily`/`unattended-upgrades` and forces apt IPv4
  for the current Vz userspace egress helper; these are documented Vz slice
  assumptions until shared image convergence either applies them to CH too or
  removes/narrows them with explicit validation
- first-contact auto-provisioning defaults to `MOTLIE_VZ_INLINE_VALIDATION=0`
  when launched through `libs/vmm`
- `MOTLIE_VZ_CONTROL_READY_FILE` is the `interactive-ready` gate, not the full
  certification gate
- missing base-image packages or a missing `/usr/local/bin/motlie-vfs-guest`
  are contract violations, not reasons to run `apt-get`, `rustup`, or `cargo`
  in the first SSH path
- missing `/usr/local/bin/codex` or `/usr/local/bin/claude` are contract
  violations, not reasons to repair npm global symlinks during first SSH
- missing or stale base-image service units, SSH CA config, profile scripts, or
  `MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V3` are contract violations; rebuild
  the v1.45 base image instead of copying scripts during first SSH
- missing prebuilt host Vz runner or egress helper artifacts are launch
  prerequisites, not reasons to run host `cargo build` in first-contact startup
- Vz no longer creates a per-guest seed DMG by default for the transitional
  mount-config path, and no longer uploads a tar seed by default; the only
  default post-boot payload is inline mount YAML
- `MOTLIE_VZ_USE_SEED_DISK=1` is retained only as an opt-in
  diagnostic/comparison path until seed/cloud-init delivery is converged
- full validation remains available through the harness `validate <guest>`
  command and saved scenarios

The current Vz path still contains post-boot per-guest mutation for identity,
CA principal wiring, VFS mount config wiring, VFS service restart, and
agent-state bind setup. That is a known transition state. The next convergence
work is to move those pieces into cloud-init, seed disk content, or base-image
content until CH and Vz share the same bootup long poles.

Implementation discipline:

- Start from the v1.4 CH contract when adding or repairing Vz behavior.
- Do not add convenience fallback work to guest startup just because it helps a
  standalone smoke script recover from a stale image.
- If the image is missing immutable content, fail fast and rebuild the image.
- If Vz requires a platform-specific seed mechanism, profile it separately and
  document why it cannot match the CH overlay/cloud-init path before making it
  part of first-contact SSH.

## Work Queue

Near-term v1.45 tasks:

- move per-guest user, uid, gid, sudoers, password, and SSH CA principal setup
  into cloud-init or the seed consumed by cloud-init
- render VFS mount config and unit drop-ins as seed/base-image content instead
  of copying and installing them over SSH
- move agent-state setup to a systemd/cloud-init unit that runs before
  `interactive-ready`
- make `interactive-ready` record phase timings and surface contract failures
  in `vz-launch-result.json`
- keep `validate <guest>` as the explicit certification path for egress, CLI,
  and package-manager checks

Future VMM/VFS/VNET convergence tasks:

- make `libs/vmm/docs/DESIGN_GUEST_IMAGE.md` the canonical design for the
  v1.5 common guest image, guest binary packaging, and VMM-owned guest
  agent location
- define one backend-neutral guest seed schema for CH and Vz
- make image builders produce the same immutable guest contract regardless of
  hypervisor
- add the canonical v1.5 guest mounter under `libs/vmm/bins/v1.5` with
  reusable guest runtime code under `libs/vmm/src/guest`, rather than forking
  historical v1.1/v1.15 binaries or creating VFS/VNET v1.5 trees
- keep VFS and VNET changes limited to reusable library bug fixes or
  functionality-gap fixes required by the VMM-owned guest runtime
- make VFS readiness report backend-neutral mount states rather than
  backend-specific service side effects
- make VNET readiness report backend-neutral egress states and keep
  certification probes out of first SSH
- converge launcher/result artifacts so every backend emits the same phase and
  validation fields

## Manual Verification Rule

For manual handoff, do not treat first SSH as certification. The required
sequence is:

1. boot or auto-provision the guest
2. wait for `interactive-ready`
3. SSH in if the goal is interactive use
4. run `validate <guest>` or a saved scenario before claiming full parity
