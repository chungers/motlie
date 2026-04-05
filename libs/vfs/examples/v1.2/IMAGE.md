# v1.2 Guest Image Notes

`v1.2` is still in active Phase 3 implementation.

The authoritative design/status document for this example is
[README.md](./README.md). Use that file first.

What is already true about the `v1.2` image build:

- it still builds one generic shared base image set under `artifacts/base/`
- guest identity remains launch-time state, not build-time state
- guest-local root writes still live in the per-launch writable ext4 overlay
- the image now includes:
  - `systemd-networkd` enabled for the egress NIC
  - validation packages such as `curl`, `dnsutils`, and `ca-certificates`
  - login-time agent-state redirection into the dedicated read/write
    VFS-backed `/agent-state` layer

What is not yet fully documented here:

- the final composed `v1.2` host flow once `motlie-vnet` is launched directly
  from the control plane
- the final end-to-end validation runbook for internet access and CLI auth

Until that composed flow is validated, prefer [README.md](./README.md) for the
current implementation status and use the forked `repl_host_v1_2` /
`launch-ch.sh` flow as the only runtime entry point.
