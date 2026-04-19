# motlie-vfs Apple Vz Transport Design

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-19 | @vmm-vz-cdx | Record the native Apple Vz `v1.15` transport shape, clarify Tart's reduced role to base-image scaffolding, and document the current signed-helper / virtio-socket operator flow |

## Scope

This document records the Apple Vz-specific guest filesystem transport decision
for `motlie-vfs`.

It exists to keep the VFS-layer design separate from the broader `libs/vmm`
backend design in `libs/vmm/docs/DESIGN_VZ.md`.

## Current State

`libs/vfs/examples/v1.15` is the first Apple Vz managed guestfs feasibility
slice.

What `v1.15` now implements:

- a Linux guest provisioned for Apple Vz can run `motlie-vfs-guest-v1_15`
- a host-side `FsServer` can serve Alice and Bob simultaneously
- multi-tag routing via `TAG <name>` still works
- the guest path is now AF_VSOCK and the host path is now Apple
  `VZVirtioSocketListener` -> Unix socket bridging

What `v1.15` now uses:

- Tart to provision the reusable runtime disk image and guest-side services
- a small Apple Vz helper to boot that same disk with
  `VZVirtioSocketDeviceConfiguration`
- guest AF_VSOCK to host Unix-socket bridging, matching the CH transport shape

What is still blocked on this unsigned development host:

- completing Apple Vz VM startup through the helper without
  `com.apple.security.virtualization`

## CH Baseline

For Linux CH/KVM, `motlie-vfs` does **not** use TAP as its filesystem
transport.

The current CH path is:

- guest `motlie-vfs-guest` connects over virtio-vsock
- Cloud Hypervisor bridges the guest vsock connection onto a host Unix socket
- `FsServer` serves the framed `FsOp` / `FsResult` protocol over that stream

TAP belongs to guest networking, not guestfs transport.

That distinction matters because Vz should converge on the same architectural
shape: a hypervisor-local side channel for filesystem traffic, not general guest
networking.

## Implemented Vz Transport Direction

The `v1.15` runtime transport is Apple virtio-socket, not Tart NAT TCP.

Virtualization.framework exposes a socket device API through:

- `VZVirtioSocketDeviceConfiguration`
- `VZVirtioSocketDevice`
- `VZVirtioSocketListener`
- `VZVirtioSocketConnection`

This is the closest Apple Vz equivalent to the CH/KVM virtio-vsock transport.

The implemented `v1.15` shape is:

- the guest continues to speak the existing framed `FsOp` / `FsResult`
  protocol
- the guest opens one connection per mount/tag over the Vz virtio-socket device
- a small Objective-C helper owns the host-side `VZVirtioSocketListener`
- that helper bridges each accepted `VZVirtioSocketConnection` onto the same
  host Unix socket path consumed by the existing `VsockConnectionHandler`
- the filesystem transport remains local to the hypervisor boundary and does
  not depend on guest networking

## Why Tart Still Exists In `v1.15`

Tart still appears in `v1.15`, but only as a provisioning tool:

- it provides a signed launcher and guest agent for one-time guest provisioning
- it lets the example reuse the same bootable raw disk image it already
  manages under `~/.tart/vms/<name>/disk.img`
- after provisioning, Tart is stopped and the Apple Vz helper launches that
  same disk image for the real virtio-socket-backed runtime

What `v1.15` no longer uses at runtime:

- guest TCP over the Tart NAT gateway
- host TCP listeners in `motlie-vfs`

## Implications For Follow-up Work

`v1.15` should be read as:

- first implementation of the intended Apple virtio-socket transport
- transport handoff validated up to the Apple entitlement gate
- still not the final architectural endpoint because the helper is example-local
  rather than the shared `libs/vmm` `vz-runner`

So the next transport-focused work should move from:

- example-local `vz-vsock-runner.m`

to:

- the shared `libs/vmm` `vz-runner`

without changing the higher-level `FsServer` / tag-routing model.

## External References

- Apple Virtualization sockets overview:
  https://developer.apple.com/documentation/virtualization/sockets
- `VZVirtioSocketListenerDelegate`:
  https://developer.apple.com/documentation/virtualization/vzvirtiosocketlistenerdelegate
- `VZVirtioSocketDevice.connect(toPort:)`:
  https://developer.apple.com/documentation/virtualization/vzvirtiosocketdevice/connect%28toport%3Acompletionhandler%3A%29
- VM runtime `socketDevices`:
  https://developer.apple.com/documentation/virtualization/vzvirtualmachine/socketdevices
