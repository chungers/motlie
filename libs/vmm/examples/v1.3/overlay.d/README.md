# Optional Overlay Content

`launch-ch.sh` copies these trees into the per-guest runtime ext4 overlay if they exist:

- `overlay.d/common/`
- `overlay.d/alice/`
- `overlay.d/bob/`

This is not a third deployed image layer. These directories are only host-side
seed content for the launch-time writable overlay.

Each directory is copied into the overlay `upper/` tree as-is, so paths are
interpreted relative to the guest root.

Examples:

```text
overlay.d/common/usr/local/bin/hello-v11
overlay.d/alice/etc/profile.d/alice-demo.sh
overlay.d/bob/opt/tools/bob-helper
```

This makes it practical to add standalone scripts or self-contained binaries
without rebuilding the shared squashfs root image.

Caveats:

- Ownership and mode come from the seed tree on the host.
- For `/usr/local/bin`, executable scripts and self-contained binaries are the
  best fit.
- If you need package-managed software, shared libraries not already present in
  the base rootfs, or stricter root-owned system state, rebuild the shared base
  image instead of relying on the overlay.
