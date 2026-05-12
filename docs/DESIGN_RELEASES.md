# Release Distribution Design

## Changelog

- 2026-04-29, @gpt55-dgx: Initial release distribution design for Motlie native binaries, npm packages, direct installers, and macOS Homebrew distribution.
- 2026-04-29, @gpt55-dgx: Updated Homebrew tap target to `motlie/homebrew-tap` and clarified installer script hosting plus archive/npm install modes.
- 2026-04-29, @gpt55-dgx: Added macOS code-signing and installed-path execution requirements for npm, direct installer, and Homebrew flows.
- 2026-05-12, @gpt55-dgx: Generalized the design around a user-specified binary target; `mmux` is now the first worked validation target only.

## Status

Draft for issue #234, `[Releases] Multi-architecture / platform binary distribution`.

This is greenfield product packaging design. There is no backwards-compatibility or migration requirement unless a later release process introduces one.

## Problem

Motlie needs a repeatable way to distribute native binaries across macOS and Linux for multiple CPU architectures. The first worked validation target is `mmux`, but the distribution model is parameterized by binary and must support future Motlie binaries, including possible model-related binaries with CUDA-enabled builds.

The distribution contract must make platform-specific assets precise without making the user experience feel platform-specific. Users should still run stable command names such as the requested `<bin>`.

## Goals

- Publish native Motlie binaries for macOS and Linux on x64 and arm64.
- Support direct GitHub Release archives, npm packages under the `@motlie` org, and a macOS Homebrew tap.
- Keep host/SSH-safe binaries such as `mmux` safe for host-wide SSH `ForceCommand` usage by executing the native Rust binary directly.
- Avoid Node launcher scripts in native binary runtime paths.
- Ensure macOS builds are signed or re-signed at final install location so Apple Silicon hosts can execute the binary reliably.
- Make CUDA an explicit optional accelerator suffix only for binaries that ship CUDA-enabled builds.
- Keep release metadata, artifact naming, package naming, and installer behavior driven by one shared release manifest.

## Non-Goals

- Do not require CUDA for binaries that do not publish CUDA-enabled functionality. The worked `mmux` target is CPU/default only.
- Do not put CUDA, GPU, Node, npm lifecycle scripts, or model runtime dependencies in the SSH login path.
- Do not make Homebrew solve Linux packaging.
- Do not require deb, rpm, or container images in the first release implementation.
- Do not publish a JavaScript launcher as a native Motlie binary command.

## Release Target Model

The release framework is generic. Each release target must define the binary-specific fields before implementation or publication begins.

```text
BIN=<installed command name>
CARGO_PACKAGE=<cargo package name>
CARGO_BIN=<cargo binary name>
VERSION=<release version>
INSTALL_PATH=<default absolute install path, if any>
FORMULA=<Homebrew formula name, if Homebrew is enabled>
NPM_PREFIX=@motlie/<package-prefix>
INSTALLER=install-<bin>.sh
FORCE_COMMAND_SAFE=true|false
```

Worked `mmux` target:

```text
BIN=mmux
CARGO_PACKAGE=motlie-mmux
CARGO_BIN=mmux
VERSION=0.1.0
INSTALL_PATH=/usr/local/bin/mmux
FORMULA=mmux
NPM_PREFIX=@motlie/mmux
INSTALLER=install-mmux.sh
FORCE_COMMAND_SAFE=true
```

## User Experience

Users run the requested command regardless of distribution channel:

```sh
<bin> --help
<bin> --version
```

Platform, architecture, libc, and accelerator names belong in archive and package names, not in the installed executable name. `mmux` examples below are worked examples for the first validation target.

### npm

The npm channel is best for developer and CI installs where package-manager pinning matters. Because native Motlie binaries must not use a Node boot script, npm users install the native package for their binary/platform pair.

Generic package shape:

```text
@motlie/<bin>-linux-x64-gnu
@motlie/<bin>-linux-arm64-gnu
@motlie/<bin>-darwin-arm64
@motlie/<bin>-darwin-x64
```

Worked `mmux` examples:

Linux x64 glibc:

```sh
npm install -g @motlie/mmux-linux-x64-gnu
mmux --help
```

Linux arm64 glibc:

```sh
npm install -g @motlie/mmux-linux-arm64-gnu
mmux --help
```

macOS Apple Silicon:

```sh
npm install -g @motlie/mmux-darwin-arm64
mmux --help
```

macOS Intel:

```sh
npm install -g @motlie/mmux-darwin-x64
mmux --help
```

For SSH host integration through npm, pin the native binary to a stable path:

```sh
npm install -g @motlie/mmux-linux-x64-gnu
ln -sf "$(npm bin -g)/mmux" /usr/local/bin/mmux
/usr/local/bin/mmux --version
```

Tradeoffs:

- npm gives org ownership, semver, package-manager pinning, CI cache compatibility, and npm publishing provenance.
- Native npm packages can expose the Rust binary directly.
- Users must know the right platform package unless another installer selects it.
- Global npm bin paths vary, so host-wide SSH integration still needs a stable symlink or wrapper path.
- npm cannot naturally select by CUDA availability; accelerator package selection must be explicit or handled by an installer.

macOS signing requirement:

- Darwin npm packages must contain a valid signed Mach-O. Ad-hoc signing is acceptable for this release path.
- CI must verify the installed npm command from its final npm install path, not only the build path.
- If an installer or admin copies the npm-installed binary into `/usr/local/bin`, the final copied binary must be re-signed in place.

### Direct Installer

The direct installer is the preferred UX for host administrators, especially for host/SSH-safe binary deployment. It can inspect the machine, select the correct GitHub Release archive, verify checksums, and install the native binary to a stable path.

Latest-style UX:

```sh
curl -fsSL https://motlie.github.io/install/mmux.sh | sh
mmux --help
```

Release-pinned UX:

```sh
curl -fsSL https://github.com/chungers/motlie/releases/download/v0.1.0/install-mmux.sh | sh
/usr/local/bin/mmux --version
```

Safer audit-before-run UX:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/v0.1.0/install-mmux.sh
shasum -a 256 install-mmux.sh
sh install-mmux.sh
```

Tradeoffs:

- The installer gives the best host-admin UX because users do not need to know package names or platform triples.
- The installer can detect OS, architecture, libc, and accelerator capability before selecting an artifact.
- The installer can install directly to a stable path such as `/usr/local/bin/<bin>`.
- `curl | sh` has trust and audit concerns, so docs must include release-pinned and checksum-verified alternatives.
- Installer logic needs portability tests and an upgrade/uninstall story.

macOS signing requirement:

- The installer must re-sign the final installed binary after copying it into the target prefix on macOS.
- The installer must verify the final installed path with `codesign --verify` and execute `<bin> --version` from that path.

Installer scripts may support multiple source modes:

```sh
sh install-mmux.sh --source archive
sh install-mmux.sh --source npm
sh install-mmux.sh --prefix /usr/local
```

Recommended defaults:

- Host/SSH-safe binaries should default to `--source archive` because host-wide deployment should not depend on npm or Node.
- Developer and CI workflows may use `--source npm` when package-manager state is desired.
- CUDA-capable model tools may detect CUDA and select either direct archives or npm packages, depending on `--source`.

### Homebrew

Homebrew is the preferred macOS package-manager UX. It should consume the same Motlie source release, but it should not define a separate release system.

Generic Homebrew UX:

```sh
brew tap motlie/tap
brew install <formula>
<bin> --help
```

Worked `mmux` UX:

```sh
brew tap motlie/tap
brew install mmux
mmux --help
```

or:

```sh
brew install motlie/tap/mmux
```

macOS SSH `ForceCommand` examples for host/SSH-safe binaries should use the configured `INSTALL_PATH`. Worked `mmux` examples:

```sshconfig
Match Group mmux-users
    ForceCommand /opt/homebrew/bin/mmux
    PermitTTY yes
```

Intel macOS:

```sshconfig
Match Group mmux-users
    ForceCommand /usr/local/bin/mmux
    PermitTTY yes
```

Tradeoffs:

- Homebrew gives macOS users the most familiar install and upgrade UX.
- Homebrew bottles provide prebuilt macOS binaries without requiring users to know Darwin asset names.
- The tap is macOS-only for this design; Linux remains npm/archive/installer-driven until a separate Linux package-manager strategy is chosen.
- Homebrew install prefixes differ between Apple Silicon and Intel, so SSH docs must call out both paths or recommend a stable admin-managed symlink.
- Homebrew formula tests must execute `#{bin}/<bin>` from the installed path so code-signing problems are caught before bottle publication.

## Distribution Channels

### GitHub Releases

The `chungers/motlie` repository is the canonical source release and artifact host.

Each Motlie release should publish:

- Source tag, for example `v0.1.0`.
- Native `.tar.gz` archives.
- Checksums.
- Installer scripts such as `install-<bin>.sh`; `install-mmux.sh` is the worked example.
- Release notes.

### npm

The npm registry hosts generated native packages under the `@motlie` org. Package templates and release scripts should live in the Motlie repo, but generated package directories do not need to be committed unless the PLAN chooses that workflow.

Native package example, parameterized by binary:

```json
{
  "name": "@motlie/<bin>-linux-x64-gnu",
  "version": "0.1.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "libc": "glibc",
  "files": ["bin/<bin>", "README.md", "LICENSE"],
  "bin": {
    "<bin>": "bin/<bin>"
  }
}
```

Worked `mmux` package example:

```json
{
  "name": "@motlie/mmux-linux-x64-gnu",
  "version": "0.1.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "libc": "glibc",
  "files": ["bin/mmux", "README.md", "LICENSE"],
  "bin": {
    "mmux": "bin/mmux"
  }
}
```

Do not publish a convenience package that exposes a JavaScript boot script as the native command. If a convenience package is revisited later, it must not be the documented ForceCommand path for host/SSH-safe binaries.

### Homebrew Tap

Homebrew requires an additional package repository:

```text
motlie/homebrew-tap
```

The tap should contain one formula per Homebrew-enabled binary. For the `mmux` worked example:

```text
Formula/mmux.rb
```

The formula should build from the Motlie source tag:

```ruby
class Mmux < Formula
  desc "TUI tmux session selector"
  homepage "https://github.com/chungers/motlie"
  url "https://github.com/chungers/motlie/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<source-tarball-sha256>"
  license "<repo-license>"

  depends_on "rust" => :build
  depends_on "tmux"

  def install
    system "cargo", "install", *std_cargo_args(path: "bins/mmux")
  end

  test do
    assert_match "mmux", shell_output("#{bin}/mmux --help")
  end
end
```

If the workspace layout needs a more explicit build:

```ruby
system "cargo", "build", "--release", "--locked", "-p", "motlie-mmux"
bin.install "target/release/mmux"
```

On macOS, the formula should ensure the installed binary is signed after `bin.install`:

```ruby
system "codesign", "--force", "--sign", "-", bin/"mmux" if OS.mac?
```

The formula test must run the installed binary:

```ruby
test do
  assert_match "mmux", shell_output("#{bin}/mmux --version")
end
```

Homebrew bottle naming and hosting should follow Homebrew/tap conventions. The release manifest should identify the Homebrew channel and supported targets, but it should not force npm/archive asset names onto Homebrew bottles.

## macOS Code Signing

Apple Silicon requires executable Mach-O binaries to have a valid code signature. Rust/macOS builds normally produce an ad-hoc signature, but copying a binary to a new location can expose validation problems at the final installed path. A failure can appear as an immediate `SIGKILL` on startup.

Release requirements:

- Every Darwin release artifact must contain a signed binary at `bin/<bin>`.
- Ad-hoc signing with `codesign --force --sign -` is sufficient for the first release design.
- The final installed path must be verified and executed, not just the build output path.
- Direct installers must re-sign after copying into the install prefix.
- Homebrew formulae should re-sign after `bin.install` and before tests.
- npm Darwin packages should package a signed binary; any later copy into `/usr/local/bin` must re-sign the copied file.

Expected verification commands, parameterized by binary:

```sh
codesign --force --sign - "target/release/${BIN}"
codesign --verify --strict --verbose=2 "target/release/${BIN}"
"target/release/${BIN}" --version
```

Worked `mmux` verification commands:

```sh
codesign --force --sign - target/release/mmux
codesign --verify --strict --verbose=2 target/release/mmux
target/release/mmux --version
```

After direct installation, parameterized by binary:

```sh
install -m 755 "${BIN}" "${INSTALL_PATH}"
codesign --force --sign - "${INSTALL_PATH}"
codesign --verify --strict --verbose=2 "${INSTALL_PATH}"
"${INSTALL_PATH}" --version
```

Worked `mmux` direct-install verification:

```sh
install -m 755 mmux /usr/local/bin/mmux
codesign --force --sign - /usr/local/bin/mmux
codesign --verify --strict --verbose=2 /usr/local/bin/mmux
/usr/local/bin/mmux --version
```

For npm verification on macOS, parameterized by binary:

```sh
npm install -g "@motlie/<bin>-darwin-arm64"
codesign -dvv "$(which <bin>)"
codesign --verify --strict --verbose=2 "$(which <bin>)"
<bin> --version
```

Worked `mmux` npm verification:

```sh
npm install -g @motlie/mmux-darwin-arm64
codesign -dvv "$(which mmux)"
codesign --verify --strict --verbose=2 "$(which mmux)"
mmux --version
```

For Homebrew verification, the bottle/formula test must execute the installed formula binary:

```sh
$(brew --prefix)/bin/<bin> --version
```

Worked `mmux` Homebrew verification:

```sh
$(brew --prefix)/bin/mmux --version
```

Future public distribution may add Developer ID signing and notarization. That is separate from the minimum ad-hoc signature needed for Apple Silicon execution and can be designed later if Gatekeeper/download UX requires it.

## Artifact Naming

Use a canonical archive name:

```text
motlie-{bin}-v{version}-{os}-{arch}[-{libc}][-{accelerator}].tar.gz
```

Examples:

```text
motlie-mmux-v0.1.0-darwin-arm64.tar.gz
motlie-mmux-v0.1.0-darwin-x64.tar.gz
motlie-mmux-v0.1.0-linux-x64-gnu.tar.gz
motlie-mmux-v0.1.0-linux-arm64-gnu.tar.gz
motlie-models-v0.1.0-linux-x64-gnu.tar.gz
motlie-models-v0.1.0-linux-x64-gnu-cuda-12.4.tar.gz
```

The executable inside the archive keeps the stable command name:

```text
bin/<bin>
```

## npm Package Naming

Generic package names:

```text
@motlie/<bin>-darwin-arm64
@motlie/<bin>-darwin-x64
@motlie/<bin>-linux-x64-gnu
@motlie/<bin>-linux-arm64-gnu
@motlie/<bin>-linux-x64-musl
@motlie/<bin>-linux-arm64-musl
```

Worked `mmux` package names:

```text
@motlie/mmux-darwin-arm64
@motlie/mmux-darwin-x64
@motlie/mmux-linux-x64-gnu
@motlie/mmux-linux-arm64-gnu
@motlie/mmux-linux-x64-musl
@motlie/mmux-linux-arm64-musl
```

For future CUDA-capable binaries:

```text
@motlie/{bin}-linux-x64-gnu
@motlie/{bin}-linux-x64-gnu-cuda-12-4
@motlie/{bin}-linux-arm64-gnu
@motlie/{bin}-linux-arm64-gnu-cuda-12-4
```

Archive names may use `cuda-12.4`. npm package names should use `cuda-12-4` to avoid dots in the accelerator suffix.

## Upload and Publishing Workflow

The release process uploads to multiple destinations. They are not all hosted in the Motlie source repository.

### Repository and Registry Ownership

- `chungers/motlie`: source tag, GitHub Release archives, checksums, installer scripts, and release notes.
- npm registry under `@motlie`: generated native npm packages.
- `motlie/homebrew-tap`: Homebrew formula and bottle metadata for macOS and future Motlie formulae.
- Optional GitHub Pages under `motlie.github.io`: stable latest installer URLs or documentation redirects.

### Installer Script Hosting

Installer script source should live in the Motlie source repository:

```text
release/
  install/
    install-mmux.sh
    install-motlie-models.sh
    lib/
      detect-os.sh
      detect-arch.sh
      detect-libc.sh
      detect-cuda.sh
      fetch-release-asset.sh
      verify-checksum.sh
```

The generic naming rule is `install-<bin>.sh`; the listed files are examples.

The canonical hosted installer scripts should be uploaded to version-pinned GitHub Releases:

```text
https://github.com/chungers/motlie/releases/download/v0.1.0/install-mmux.sh
https://github.com/chungers/motlie/releases/download/v0.1.0/install-motlie-models.sh
```

GitHub Pages may provide latest convenience entrypoints:

```text
https://motlie.github.io/install/mmux.sh
https://motlie.github.io/install/motlie-models.sh
```

Production and automation docs should prefer release-pinned GitHub Release URLs. Pages URLs are convenience shortcuts and should redirect to, or fetch from, the current release-pinned installer.

In npm mode, installer scripts can select platform-specific native packages instead of direct archives. For a CUDA-capable binary, npm-mode selection could choose:

```text
@motlie/models-linux-x64-gnu-cuda-12-4
```

or fall back to:

```text
@motlie/models-linux-x64-gnu
```

### Release Upload Sequence

1. Create and push the Motlie source tag.
2. Build and test native binaries from that tag.
3. Sign and verify Darwin binaries.
4. Upload canonical archives, checksums, and installer scripts to the `chungers/motlie` GitHub Release.
5. Generate native npm packages from the same build outputs.
6. Publish npm packages to the `@motlie` org.
7. Update `motlie/homebrew-tap` with the new formula version and source tarball checksum.
8. Run tap CI to build/test the formula on macOS Apple Silicon and Intel.
9. Publish Homebrew bottles using the tap's bottle workflow and update the formula bottle block if required.
10. Run install verification for npm, direct installer, and Homebrew from final installed paths.

### Publishing Credentials

The DESIGN should prefer trusted publishing or short-lived CI credentials where supported. If manual tokens are required for the first release, the PLAN must isolate them to release workflows and avoid storing them in the source tree.

## Release Manifest

A shared manifest should drive artifact, package, installer, and Homebrew metadata.

Example manifest entries. The `mmux` entry is the worked validation target; `motlie-models` illustrates a future non-ForceCommand target with a CUDA-capable variant.

```toml
[release]
version_source = "workspace"

[[binary]]
name = "mmux"
cargo_package = "motlie-mmux"
cargo_bin = "mmux"
force_command_safe = true
install_path = "/usr/local/bin/mmux"

[binary.channels.archive]
enabled = true

[binary.channels.npm]
enabled = true
scope = "@motlie"

[binary.channels.homebrew]
enabled = true
tap = "motlie/homebrew-tap"
formula = "mmux"
targets = ["darwin-arm64", "darwin-x64"]

[[binary.target]]
os = "linux"
arch = "x64"
libc = "gnu"

[[binary.target]]
os = "linux"
arch = "arm64"
libc = "gnu"

[[binary.target]]
os = "darwin"
arch = "arm64"

[[binary.target]]
os = "darwin"
arch = "x64"

[[binary]]
name = "motlie-models"
cargo_package = "motlie-models"
cargo_bin = "motlie-models"
force_command_safe = false

[[binary.target]]
os = "linux"
arch = "x64"
libc = "gnu"

[[binary.target]]
os = "linux"
arch = "x64"
libc = "gnu"
accelerator = "cuda-12.4"
```

The PLAN should decide whether this lives at `release/motlie-release.toml` or `motlie-release.toml`.

## Core Motlie Work

The first release implementation should avoid adding runtime dependencies to core Motlie libraries. Some shared helpers may still be justified.

Candidate work:

- Common build metadata for `--version` and possibly `--version --json`.
- Runtime self-check helpers for external dependencies such as `tmux`, `ssh`, CUDA libraries, or model assets.
- Artifact and package naming helpers driven by the release manifest.
- A release manifest parser in an internal `xtask` or release helper crate.
- Shell installer platform detection, ideally generated or validated from the release manifest.
- macOS signing helpers for release packaging and installer flows.

Initial recommendation:

- Implement release automation in an internal `xtask` or `release/` tool first.
- Add runtime library helpers only when multiple binaries need the same metadata/self-check behavior.
- Keep CUDA/model dependencies out of non-CUDA binary targets such as the worked `mmux` target.

## Alternatives Considered

### npm Only

Simple for developers and CI, but poor for host-wide SSH because platform package names are exposed and global npm bin paths vary.

### Installer Only

Best host-admin UX, but loses package-manager state and puts more trust in shell installer logic.

### Homebrew First

Excellent macOS UX, but does not cover Linux and does not replace npm/archive distribution.

### Prebuilt Archives in Homebrew Formula

This aligns Homebrew directly with Motlie archive names, but a source formula plus bottles is more idiomatic for Homebrew. Keep prebuilt archive formulae as a fallback for the custom tap.

## Open Questions

- Should the first implementation publish only the `mmux` worked target, then enable other binary targets after the workflow is proven?
- Are Linux musl builds required in v0.1?
- Should direct installer scripts be hosted on GitHub Releases only, GitHub Pages only, or both?
- Should installer platform/CUDA detection be handwritten shell, generated from a manifest, or implemented in a helper binary?
- Should every Motlie binary support `--version --json` and `--self-check`?
- Should macOS public-download releases add Developer ID signing and notarization after the ad-hoc-signing release path is working?
- Should release automation live in `xtask`, `release/`, or a new internal release helper crate?

## References

- Issue #234: https://github.com/chungers/motlie/issues/234
- npm package metadata: https://docs.npmjs.com/cli/v11/configuring-npm/package-json/
- npm scoped packages: https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/
- Homebrew Formula Cookbook: https://docs.brew.sh/Formula-Cookbook
- Homebrew Taps: https://docs.brew.sh/Taps
- Homebrew Bottles: https://docs.brew.sh/Bottles
