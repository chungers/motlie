# v1.25 Image Convergence Notes

`v1.25` should start converging the CH and Apple Vz image lines rather than
creating another permanently divergent guest contract.

## Immediate Goal

Align the CH `v1.2` and Apple Vz `v1.25` images on:

- packages
- users
- uid / gid targets
- passwords
- locale
- shell/profile behavior
- systemd units
- agent-state layout

The target is one logical guest image definition, even if the platform-specific
boot artifacts still differ.

## What May Stay Backend-Specific At First

- kernel / boot artifact shape
- CH squashfs + overlay mechanics
- Apple Vz disk-image boot mechanics
- launch-time attachment and provisioning details

Those differences are acceptable early in `v1.25` as long as the guest-visible
contract converges.

## Current Practical Starting Point

Today the two proven slices differ:

- CH `v1.2` image is directly built under `libs/vnet/examples/v1.2`
- Apple Vz `v1.15` uses a Tart-backed Ubuntu base and native post-boot
  provisioning

So the first useful `v1.25` image work is not “one file for both.” It is:

- make the Apple Vz guest look more like the CH `v1.2` guest from inside
- remove avoidable uid/gid/password/package drift

## Expected First Convergence Work

- align Alice/Bob uid/gid targets with the `v1.2` line
- align guest passwords
- align installed validation/dev packages
- align MOTD/profile behavior where it helps the demo
- align `/agent-state` redirection behavior

Once that is working, we can decide whether the CH and Vz image-build logic can
share more of the actual build pipeline.

