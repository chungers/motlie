# `v1.05` Apple Vz Tart Boot Probe

This directory is the first Apple Vz feasibility slice for `libs/vmm`.

It is intentionally separate from the existing Cloud Hypervisor examples:

- it does not modify `v1.3` or `v1.4`
- it does not replace any CH launch path
- it exists only to prove that a Linux guest can boot on macOS through Apple
  `Virtualization.framework` using Tart as the temporary signed launcher

The purpose of `v1.05` is narrower than the later Vz slices:

- prove Apple Vz guest boot works on the host
- prove a guest gets an IP over the default NAT path
- prove guest command execution works
- measure time from host launch to guest readiness

It is not yet the full Motlie Vz backend and it does not attempt to exercise
`motlie-vfs` or `motlie-vnet`.

## Files

| File | Purpose |
|------|---------|
| `tart-vz-smoke.sh` | Clone, launch, validate, and stop a Tart-backed Linux guest without touching the CH path |

## Requirements

- macOS on Apple Silicon
- Tart installed and working
- no `sudo`

The script uses:

- Tart's default shared/NAT networking
- a prebuilt Ubuntu Linux VM image from `ghcr.io/cirruslabs/ubuntu:latest`
- `tart exec` for in-guest validation

## Quick Start

```bash
./tart-vz-smoke.sh
```

This will:

1. clone `ghcr.io/cirruslabs/ubuntu:latest` to a local VM named `motlie-v1-05-ubuntu`
2. start it headlessly with Tart
3. poll until Tart reports a guest IP
4. run `uname -a` inside the guest
5. write a JSON result under `artifacts/result.json`
6. stop the guest

## Outputs

The script writes:

- `artifacts/tart-run.log`
  - launcher stdout/stderr
- `artifacts/result.json`
  - boot timing and guest validation summary

## Why Tart Here

For `v1.05`, Tart is a temporary Vz launcher:

- it already solves the app-bundle signing / entitlement story
- it proves Linux guest boot through Apple Vz without requiring Motlie's own
  `vz-runner` to be signed yet

This does not change the long-term design:

- the final backend still targets a Motlie-owned `vz-runner`
- this directory exists only to answer the host feasibility question quickly
