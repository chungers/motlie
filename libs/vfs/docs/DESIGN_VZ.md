# motlie-vfs Apple Vz Transport Design

## Scope

This document records the Apple Vz-specific guest filesystem transport decision
for `motlie-vfs`.

It exists to keep the VFS-layer design separate from the broader `libs/vmm`
backend design in `libs/vmm/docs/DESIGN_VZ.md`.

## Current State

`libs/vfs/examples/v1.15` is the first Apple Vz managed guestfs feasibility
slice.

What `v1.15` proved:

- a Linux guest booted through Apple Vz on this host can run
  `motlie-vfs-guest-v1_15`
- a host-side `FsServer` can serve Alice and Bob simultaneously
- multi-tag routing via `TAG <name>` still works
- managed guestfs semantics survive the Apple Vz path

What `v1.15` used:

- Tart as an interim signed Vz launcher
- guest TCP over the Tart NAT gateway back to a host TCP listener

That transport was chosen only to prove filesystem feasibility quickly.

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

## Final Vz Transport Direction

The intended final Apple Vz transport is Apple virtio-socket, not Tart NAT TCP.

Virtualization.framework exposes a socket device API through:

- `VZVirtioSocketDeviceConfiguration`
- `VZVirtioSocketDevice`
- `VZVirtioSocketListener`
- `VZVirtioSocketConnection`

This is the closest Apple Vz equivalent to the CH/KVM virtio-vsock transport.

The expected final shape is:

- the guest continues to speak the existing framed `FsOp` / `FsResult`
  protocol
- the guest opens one connection per mount/tag over the Vz virtio-socket device
- `vz-runner` owns the host-side `VZVirtioSocketListener`
- `vz-runner` bridges each accepted `VZVirtioSocketConnection` into the Motlie
  VFS protocol handler
- the filesystem transport remains local to the hypervisor boundary and does
  not depend on guest networking

## Why Tart NAT Is Not Final

Using guest TCP over the Tart NAT gateway is acceptable for `v1.15` because it:

- proved the managed guestfs semantics are portable
- avoided blocking on the first Swift/ObjC `vz-runner` integration
- kept the Vz slice narrow enough to answer feasibility quickly

But it is not the intended final design because it:

- couples filesystem reachability to guest networking
- is architecturally weaker than the CH/KVM vsock boundary
- makes filesystem transport look like ordinary guest network traffic
- would blur the line between `motlie-vfs` and `motlie-vnet`

## Implications For Follow-up Work

`v1.15` should be read as:

- feasibility proven for managed guestfs semantics on Apple Vz
- transport decision still open at the implementation level
- likely final transport already clear at the design level: Apple virtio-socket

So the next transport-focused work should move from:

- Tart NAT TCP bridge

to:

- Vz virtio-socket listener and connection handling inside `vz-runner`

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
