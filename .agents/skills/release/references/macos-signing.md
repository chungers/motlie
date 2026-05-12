# macOS Signing

Use this reference when validating Darwin artifacts or installer behavior.

Parameterize examples by the release target:

```text
BIN=<installed command name>
INSTALL_PATH=<final installed path, e.g. /usr/local/bin/<bin>>
```

Minimum first-release requirement:

- ad-hoc sign with `codesign --force --sign -`
- verify with `codesign --verify --strict --verbose=2`
- execute from the final installed path

Build-path verification:

```sh
codesign --force --sign - "target/release/${BIN}"
codesign --verify --strict --verbose=2 "target/release/${BIN}"
"target/release/${BIN}" --version
```

Final-path verification:

```sh
sudo install -m 755 "bin/${BIN}" "${INSTALL_PATH}"
sudo codesign --force --sign - "${INSTALL_PATH}"
codesign --verify --strict --verbose=2 "${INSTALL_PATH}"
"${INSTALL_PATH}" --version
```

Why this gate exists:

- Apple Silicon validates executable signatures at startup.
- A copied Mach-O may fail at the final path even if the build-path binary runs.
- Failure can appear as immediate `SIGKILL`.

Release rule:

- Linux may cross-compile Darwin artifacts.
- Darwin artifacts are not final until the macOS signing gate signs, verifies, repacks, and executes them.

Developer ID signing and notarization are later hardening steps, not the minimum first-release gate.
