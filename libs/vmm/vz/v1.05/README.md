# Apple Vz `v1.05` Host Probe

This directory contains the narrowest possible Apple `Virtualization.framework`
probe for the Vz guest-image gate:

- `minivz.swift`
  - intended Swift implementation using `VZLinuxBootLoader`
- `minivz_objc.m`
  - Objective-C fallback used to complete host feasibility checks when the
    local Swift toolchain does not match the installed Apple SDK
- `minivz.entitlements`
  - required entitlement plist with `com.apple.security.virtualization`
- `run-probe.sh`
  - build, sign, and run wrapper

## Probe Inputs

The probe expects:

- a Linux kernel image
- an initramfs image
- a serial log output path

The current local test used Alpine Linux arm64 netboot artifacts:

- `vmlinuz-virt`
- `initramfs-virt`

with kernel command line:

- `console=hvc0 rdinit=/bin/sh printk.devkmsg=on`

That path is intentionally root-disk-free. It exists only to prove that:

- `Virtualization.framework` can boot an aarch64 Linux kernel on Apple Silicon
- the guest reaches serial-console output
- the initramfs can drop directly into a shell

## Local Findings On This Host

Host tested:

- Apple Silicon Mac mini (`Mac16,11`)
- macOS `15.5 (24F74)`

Observed blockers:

1. Swift toolchain / SDK mismatch
- `swiftc` on this host is newer than the installed Command Line Tools SDK.
- The Swift probe fails before build with:
  - `this SDK is not supported by the compiler`
- As a result, the Swift source is present, but the live host probe currently
  has to fall back to Objective-C unless a matching Xcode toolchain is installed.

2. Sandbox execution blocks Vz availability
- Running the probe in the sandbox yields:
  - `Invalid virtual machine configuration. Virtualization is not available on this hardware.`
- This is not a real hardware limitation on Apple Silicon; it reflects the
  sandboxed execution context.

3. Ad-hoc signing is not enough for the virtualization entitlement
- The binary can be ad-hoc signed with `minivz.entitlements`.
- The code signature shows the entitlement embedded.
- But outside the sandbox, `Virtualization.framework` still rejects the process:
  - `The process doesn’t have the “com.apple.security.virtualization” entitlement.`
- `security find-identity -p codesigning -v` on this host returned:
  - `0 valid identities found`

## Practical Conclusion

This probe code is sufficient to exercise the real Vz boot path, but this host
is not yet prepared to pass the gate because it lacks:

- a matching Apple Swift/Xcode toolchain for the Swift implementation
- a usable Apple code-signing identity that can grant the virtualization
  entitlement at runtime

Before the probe can produce a successful boot log, the host needs:

1. full Xcode or a matching Apple toolchain/SDK pair
2. a valid Apple Development signing identity
3. execution outside the sandbox

Once those are in place, the expected next validation is:

1. build and sign `minivz`
2. boot Alpine `vmlinuz-virt` + `initramfs-virt`
3. record time-to-first-console-output
4. capture the initramfs shell prompt on `hvc0`
