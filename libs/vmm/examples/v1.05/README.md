# `v1.05` Apple Vz Tart Guest Build / Boot Probe

This directory is the first Apple Vz feasibility slice for `libs/vmm`, but its
goal is not merely “can any Linux VM boot?”.

`v1.05` should be the Apple Vz parallel of
[`libs/vfs/examples/v1`](../../../vfs/examples/v1/README.md) as closely as
possible without changing the working Cloud Hypervisor path.

It is intentionally separate from the existing Cloud Hypervisor examples:

- it does not modify `v1.3` or `v1.4`
- it does not replace any CH launch path
- it exists to build a Motlie-controlled Linux guest on macOS through Apple
  `Virtualization.framework` using Tart as the temporary signed launcher

## `v1.05` Target

`v1.0` on the CH side tried to show:

- a Motlie-controlled guest image
- the `motlie-vfs-guest` binary inside the guest
- the guest-side `mounts.yaml` contract
- the guest systemd service shape
- a real host/guest launch and validation flow

`v1.05` should mirror that as far as possible for Apple Vz before the later
`v1.15` guestfs transport slice.

That means `v1.05` should prove:

- Apple Vz guest boot works on the host
- a Motlie-controlled Linux guest image can be prepared on top of a Tart base
- `motlie-vfs-guest` can be installed and validated inside the guest
- `/etc/motlie-vfs/mounts.yaml` can be provisioned inside the guest
- the guest-side systemd service shape can be installed
- guest command execution works
- a guest gets an IP over the default NAT path
- boot timing is measurable

`v1.05` is still not the full Motlie Vz backend. The specific `motlie-vfs`
transport proof remains the job of `v1.15`.

## Current Gaps Relative To `v1.0`

The remaining gaps between the current Vz slice and `v1.0` are:

- no host-side `motlie-vfs` server in the loop yet
- no Vz guestfs transport proof yet
- no mounted filesystem semantics yet
- no overlay semantics yet
- no policy engine semantics yet

This is deliberate sequencing:

- `v1.05` owns guest image / guest contract / host boot viability
- `v1.15` owns managed guestfs transport feasibility

## Execution Plan

The intended `v1.05` execution order is:

1. prove Tart can boot a Linux guest on this host
2. provision a Motlie-controlled guest from the base Tart image
3. install `motlie-vfs-guest`
4. install `mounts.yaml`
5. install the `motlie-vfs-guest.service` unit
6. validate the guest contract in a repeatable script

The first step is implemented today.
The provisioning path is now implemented with guest-side source upload and
build, so `v1.05` can start validating the Motlie guest contract before the
later transport slice.

## Files

| File | Purpose |
|------|---------|
| `tart-vz-smoke.sh` | Clone, launch, validate, and stop a Tart-backed Linux guest without touching the CH path |
| `build-tart-guest.sh` | Provision a Motlie-controlled guest on top of the Tart Ubuntu base image and validate the guest-side contract |

## Requirements

- macOS on Apple Silicon
- Tart installed and working
- no `sudo`

The script uses:

- Tart's default shared/NAT networking
- a prebuilt Ubuntu Linux VM image from `ghcr.io/cirruslabs/ubuntu:latest`
- `tart exec` for in-guest validation
- guest-side source upload for provisioning

## Quick Start

```bash
./tart-vz-smoke.sh
```

This will do the narrow host feasibility proof:

1. clone `ghcr.io/cirruslabs/ubuntu:latest` to a local VM named `motlie-v1-05-ubuntu`
2. start it headlessly with Tart
3. poll until Tart reports a guest IP
4. run `uname -a` inside the guest
5. write a JSON result under `artifacts/result.json`
6. stop the guest

For the Motlie-controlled guest step:

```bash
./build-tart-guest.sh
```

This provisions a local VM image variant and validates that:

- `motlie-vfs-guest` exists in the guest
- `/etc/motlie-vfs/mounts.yaml` exists
- the `motlie-vfs-guest.service` unit exists
- the Motlie source tree can be uploaded and built inside the guest

## Current Validated Result

On the current Apple Silicon macOS host, the implemented scripts have already
proven:

- Tart can boot the Ubuntu arm64 guest under Apple Vz
- the guest reaches an IP in about `4.8s`
- `tart exec` works against the running guest
- `motlie-vfs-guest` can be built inside the guest from the Motlie source tree
- `/etc/motlie-vfs/mounts.yaml` can be installed from the `v1.0` example
- the `motlie-vfs-guest.service` unit can be installed and enabled

The latest provisioning validation writes these facts to
`artifacts/build-result.json`.

## Outputs

The script writes:

- `artifacts/tart-run.log`
  - launcher stdout/stderr
- `artifacts/result.json`
  - boot timing and guest validation summary
- `artifacts/build-result.json`
  - guest-provisioning validation summary

## Why Tart Here

For `v1.05`, Tart is a temporary Vz launcher:

- it already solves the app-bundle signing / entitlement story
- it proves Linux guest boot through Apple Vz without requiring Motlie's own
  `vz-runner` to be signed yet

This does not change the long-term design:

- the final backend still targets a Motlie-owned `vz-runner`
- this directory exists to answer the guest-image and boot-contract question
  quickly
