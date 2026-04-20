# v1.25 Apple Vz Harness

`v1.25` uses the native Apple Vz launcher model proven in
`libs/vfs/examples/v1.15`, not the CH `launch-ch.sh` runtime path.

Current harness direction:

- signed Apple `Virtualization.framework` helper
- fresh per-run guest clone
- no persistent host-visible network configuration
- clean teardown by default
- NAT/SSH available for guest provisioning and bootstrap

This is the right starting point because it already proved:

- native Apple Vz guest boot
- deterministic runner lifecycle
- clean teardown
- no dependency on Tart for runtime launch

What `v1.25` still must establish on top of that:

- image convergence with the CH `v1.2` guest contract
- guest egress internet on the Vz path
- evidence about packet visibility/control beyond Apple NAT

The runtime model should therefore be:

- reuse the `v1.15` style native runner lifecycle
- adapt the guest contract toward `v1.2`
- treat Apple NAT as bootstrap help unless and until a stronger packet path is
  demonstrated

