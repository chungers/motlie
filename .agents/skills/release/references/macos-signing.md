# macOS Signing

Use this reference when validating Darwin artifacts or installer behavior.

Minimum first-release requirement:

- ad-hoc sign with `codesign --force --sign -`
- verify with `codesign --verify --strict --verbose=2`
- execute from the final installed path

Build-path verification:

```sh
codesign --force --sign - target/release/mmux
codesign --verify --strict --verbose=2 target/release/mmux
target/release/mmux --version
```

Final-path verification:

```sh
sudo install -m 755 bin/mmux /usr/local/bin/mmux
sudo codesign --force --sign - /usr/local/bin/mmux
codesign --verify --strict --verbose=2 /usr/local/bin/mmux
/usr/local/bin/mmux --version
```

Why this gate exists:

- Apple Silicon validates executable signatures at startup.
- A copied Mach-O may fail at the final path even if the build-path binary runs.
- Failure can appear as immediate `SIGKILL`.

Release rule:

- Linux may cross-compile Darwin artifacts.
- Darwin artifacts are not final until the macOS signing gate signs, verifies, repacks, and executes them.

Developer ID signing and notarization are later hardening steps, not the minimum first-release gate.
