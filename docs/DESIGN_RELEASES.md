# Release Distribution Design

## Changelog

- 2026-04-29, @gpt55-dgx: Initial release distribution design for Motlie native binaries, npm packages, direct installers, and macOS Homebrew distribution.
- 2026-04-29, @gpt55-dgx: Updated Homebrew tap target to `motlie/homebrew-tap` and clarified installer script hosting plus archive/npm install modes.
- 2026-04-29, @gpt55-dgx: Added macOS code-signing and installed-path execution requirements for npm, direct installer, and Homebrew flows.
- 2026-05-12, @gpt55-dgx: Generalized the design around a user-specified binary target; `mmux` is now the first worked validation target only.
- 2026-05-12, @gpt55-dgx: Replaced the shared manifest concept with per-binary release manifests and a release coordination PR workflow.
- 2026-05-13, @gpt55-dgx: Added structured target status, target-specific and rollup gates, evidence schema, cargo-zigbuild default, and merge-commit coordination strategy.
- 2026-05-13, @gpt55-dgx: Fixed npm global-bin command, defined optional GitHub Pages installer updates, and added installer validation as a manifest-tracked gate.
- 2026-05-13, @gpt55-dgx: Made static musl the default Linux target policy when feasible; glibc floors are required only for gnu fallback/CUDA targets.
- 2026-05-14, @gpt55-dgx: Split universal/Darwin/Linux evidence lists, pinned the default Linux musl toolchain, and documented the musl allocator trade-off.
- 2026-05-14, @gpt55-dgx: Changed releases to branch-local calver-codename events that support multiple binaries and never merge release branches back to `main`.
- 2026-05-14, @gpt55-dgx: Clarified codename terminology, concurrent release branch handling, and installer template lifecycle.
- 2026-05-14, @gpt55-dgx: Added release notes workflow for workspace and per-binary notes, including operator prompts and publication gates.
- 2026-05-14, @gpt55-dgx: Changed per-binary release manifests to stable `releases/<bin>.toml` files discovered by scanning `releases/`, with versions stored in manifest schema and aggregate GitHub notes generated from per-binary notes.
- 2026-05-14, @gpt55-dgx: Added master tracking issue and sub-issue/sub-PR orchestration around the release branch manifest ledger.
- 2026-05-16, @vmm-cdx: Added optional VM guest-image artifact targets for native per-platform image builds, OCI payload publication, and harness evidence coordinated through the same release branch, master issue, sub-issue, and sub-PR model.
- 2026-05-16, @vmm-cdx: Reprioritized VM image artifact targets for issue #258: Apple Silicon VZ and DGX/aarch64 Linux CH build `linux/arm64` first, with Linux amd64/x86_64 CH coordinated on a later native host.
- 2026-05-17, @vmm-cdx: Updated VM guest-image target examples to reference release-input configs under `releases/vmm/v1.5/configs/`.

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
- Keep release metadata, artifact naming, package naming, and installer behavior driven by branch-local workspace and per-binary release manifests.

## Non-Goals

- Do not require CUDA for binaries that do not publish CUDA-enabled functionality. The worked `mmux` target is CPU/default only.
- Do not put CUDA, GPU, Node, npm lifecycle scripts, or model runtime dependencies in the SSH login path.
- Do not make Homebrew solve Linux packaging.
- Do not require deb, rpm, or container images in the first release implementation.
- Do not publish a JavaScript launcher as a native Motlie binary command.

## Release Target Model

The release framework is generic and release-event centered. A release event is named with a calver-codename:

```text
<YYYY-MM>-<codename>
```

The month is locked when the release branch is opened. If the release slips, the name remains unchanged. The codename is freeform and may contain hyphens; `amber-aardvark` is only a worked example, not a rule that every codename must be adjective-animal. The release branch and Git tag both use this release-event name:

```text
release/<YYYY-MM>-<codename>
<YYYY-MM>-<codename>
```

Release branches are retained release ledgers and never merge back to `main`. Source or process fixes discovered on a release branch must be cherry-picked into a normal `main` PR when they are relevant outside the release branch. The branch-local `releases/` directory can therefore contain release manifests, notes, installers, and ledger updates without polluting `main`.

Codename uniqueness is enforced by remote branch and tag names. A main-branch codename registry is intentionally not required for v0 because it would reintroduce release-event state into `main`. If later automation needs a registry, it should be generated from remote refs or kept outside the source tree rather than becoming a required release artifact on `main`.

Release events may run concurrently as independent release branches. Each branch is self-contained; fixes cherry-picked to `main` from one release branch can flow into another concurrent release branch through a normal merge from `main` when relevant.

The release skill may bootstrap a release event after human confirmation. It can suggest calver-codename candidates, check remote branch/tag conflicts, create the release branch from `main`, generate initial `releases/manifest.toml`, generate one stable per-binary manifest and note file per selected native binary, add optional artifact manifests when those targets are in scope, push the branch, and create a master tracking GitHub issue. The master issue coordinates human/agent handoffs, but it does not replace the release branch manifests as the authoritative ledger.

Scoped platform/channel work should be represented as sub-issues and sub-PRs targeting the release branch. Each sub-issue identifies the binary, gate, target id, required host/platform, release branch, expected evidence, and manifest fields to update. Each sub-PR updates branch-local manifest state and closes its sub-issue when merged. A later agent can resume by reading the master issue, sub-issue/PR state, and the release branch manifests; if those sources disagree, the manifests win.

## Optional VM Image Artifact Targets

The same release-branch model also applies when a release event includes
Motlie VM guest image artifacts. VM images are optional artifact targets; a
release event may publish only native binaries, only VM image/OCI payloads, or
both. Native binary releases do not require VM image generation or VM image
gates unless those artifact targets are explicitly in scope for that release
event.

VM image artifact targets have the same coordination problem as native binary
targets: different hosts must build and validate different platform artifacts,
then one retained branch and one master issue must record what was built,
which source digest it used, which backend consumed it, and which harness
scenarios passed.

VM image artifact targets use stable manifest files under `releases/` with a
different kind from native binaries:

```text
kind = "motlie.vm-image-artifact"
```

The manifest represents one logical guest contract, for example
`motlie-vmm-guest-v1.5`, and a target matrix such as:

```text
vz-darwin-arm64    # priority 1: Apple Silicon VZ, guest linux/arm64
ch-linux-arm64     # priority 1: DGX/aarch64 Linux CH, guest linux/arm64
ch-linux-amd64     # priority 2: coordinated x86_64/amd64 Linux CH host
```

Each target records both the builder host requirement and the guest platform:

```toml
[[target]]
id = "vz-darwin-arm64"
backend = "vz"
host_os = "darwin"
host_arch = "arm64"
guest_platform = "linux/arm64"
image_config = "releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml"
builder = "mbuild"
package_engine = "native-vz-bootstrap" # VZ-native equivalent or adapter path
oci_payload_tag = "motlie-guest:v1.5-arm64"

[[target]]
id = "ch-linux-arm64"
backend = "ch"
host_os = "linux"
host_arch = "arm64"
guest_platform = "linux/arm64"
image_config = "releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-arm64.yaml"
builder = "mbuild"
package_engine = "native-ch-bootstrap" # preferred native engine once implemented
oci_payload_tag = "motlie-guest:v1.5-arm64"

[[target]]
id = "ch-linux-amd64"
backend = "ch"
host_os = "linux"
host_arch = "amd64"
guest_platform = "linux/amd64"
image_config = "releases/vmm/v1.5/configs/motlie-image.ubuntu-24.04.linux-amd64.yaml"
builder = "mbuild"
package_engine = "native-ch-bootstrap"
oci_payload_tag = "motlie-guest:v1.5-amd64"
```

For issue #258, native builders are the acceptance path. The priority order is
Apple Silicon VZ (`vz-darwin-arm64`) and DGX/aarch64 Linux CH
(`ch-linux-arm64`) first, then coordinated x86_64/amd64 Linux CH
(`ch-linux-amd64`) on a separate host. OCI platform names use `linux/amd64`;
Rust uses `x86_64-unknown-linux-musl`; Debian package architecture uses
`amd64`; Linux hosts commonly report `x86_64`.

The current `mbuild`
emitter can import an OCI source image, build static musl Motlie guest binaries
for the requested guest platform, run rootless-`chroot` package staging, emit
CH artifacts, and export a local OCI image layout. That rootless-`chroot`
engine requires qemu-user/binfmt for cross-architecture package staging.

The outstanding CH improvement is a `native-ch-bootstrap` package engine:
when the build host architecture matches the requested guest platform, CH can
boot a temporary provisioning VM and run package/setup steps inside the guest
instead of using host `chroot`. CH uses KVM and does not emulate another CPU
architecture, so this is a native same-architecture builder path. qemu-user
and binfmt remain optional cross-architecture conveniences for local iteration,
not required release gates.

VM image target gates mirror binary release gates but record image-specific
evidence:

```toml
[[gate]]
id = "oci-source-resolved"
target_id = "vz-darwin-arm64"
state = "planned"
evidence = []

[[gate]]
id = "guest-image-built"
target_id = "vz-darwin-arm64"
state = "planned"
evidence = []

[[gate]]
id = "oci-payload-exported"
target_id = "vz-darwin-arm64"
state = "planned"
evidence = []

[[gate]]
id = "harness-validated"
target_id = "vz-darwin-arm64"
state = "planned"
evidence = []
```

Evidence must include:

- build host OS/architecture and toolchain identity
- `mbuild` source commit
- source OCI image-index digest
- selected source platform-manifest digest
- assembled rootfs size and sha256
- emitted backend artifact digests
- local or published OCI layout/image manifest digest
- harness scenario result paths or PR/issue comment links
- pass/fail status for every required v1.5 scenario

The release branch stores manifests, notes, command evidence, and final
published URLs. It must not store built VM images, rootfs tarballs, disk images,
or OCI blobs in git. Those artifacts live in GitHub Release assets, registry
storage, workflow artifacts, or explicit staging storage, with checksums and
URLs recorded in the manifest.

Reusable source fixes, docs, and builder improvements discovered while working
on VM image artifact targets follow the existing rule: cherry-pick them back
to normal `main` or feature PRs. The retained release branch is a ledger and
coordination surface, not a branch that merges back into `main`.

When a release event contains native binary targets, each binary target must define the binary-specific fields before implementation or publication begins.

```text
RELEASE_NAME=<YYYY-MM-codename>
RELEASE_BRANCH=release/<release-name>
RELEASE_TAG=<release-name>
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
```

Per-binary manifests are discovered by scanning `releases/*.toml` and selecting files with `kind = "motlie.binary-release"`. Optional VM image artifact manifests are discovered by selecting files with `kind = "motlie.vm-image-artifact"`. Do not encode the release version in the manifest filename. The version belongs in the manifest identity fields, and each manifest points to its own release-note source when release notes are required.

```text
BINARY_MANIFEST=releases/<bin>.toml
BINARY_NOTES=releases/<bin>.md
```

Required fields inside each binary manifest:

```text
[identity].binary=<installed command name>
[identity].version=<release version>
[build].cargo_package=<cargo package name>
[build].cargo_bin=<cargo binary name>
[install].default_path=<default absolute install path, if any>
[install].force_command_safe=true|false
[release].notes_path=releases/<bin>.md
[archive]/[npm]/[homebrew]/[installer] explicit names as enabled
```

Worked `mmux` release target:

```text
RELEASE_NAME=2026-05-amber-aardvark
RELEASE_BRANCH=release/2026-05-amber-aardvark
RELEASE_TAG=2026-05-amber-aardvark
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
BINARY_MANIFEST=releases/mmux.toml
BINARY_NOTES=releases/mmux.md
[identity].binary=mmux
[identity].version=0.1.0
[build].cargo_package=motlie-mmux
[build].cargo_bin=mmux
[install].default_path=/usr/local/bin/mmux
[install].force_command_safe=true
[homebrew].formula=mmux
[npm].package_prefix=@motlie/mmux
[installer].script=install-mmux.sh
```

When `FORCE_COMMAND_SAFE=true`, the direct installer must default to archive mode and the login path must execute the native binary directly. When `FORCE_COMMAND_SAFE=false`, npm-mode install can be considered for non-login use cases, but package runtime paths still must execute native binaries directly unless a future design explicitly changes that contract.

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
@motlie/<bin>-linux-x64-musl
@motlie/<bin>-linux-arm64-musl
@motlie/<bin>-darwin-arm64
@motlie/<bin>-darwin-x64
```

Worked `mmux` examples:

Linux x64 static musl:

```sh
npm install -g @motlie/mmux-linux-x64-musl
mmux --help
```

Linux arm64 static musl:

```sh
npm install -g @motlie/mmux-linux-arm64-musl
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
npm install -g @motlie/mmux-linux-x64-musl
ln -sf "$(npm config get prefix)/bin/mmux" /usr/local/bin/mmux
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
curl -fsSL https://github.com/chungers/motlie/releases/download/2026-05-amber-aardvark/install-mmux.sh | sh
/usr/local/bin/mmux --version
```

Safer audit-before-run UX:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/2026-05-amber-aardvark/install-mmux.sh
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

- Source tag, for example `2026-05-amber-aardvark`.
- Native `.tar.gz` archives.
- Checksums.
- Installer scripts such as `install-<bin>.sh`; `install-mmux.sh` is the worked example.
- Release notes.

### Release Notes

Release notes are branch-local artifacts produced with the release manifests:

```text
releases/
  notes.md
  <bin>.md
```

`releases/notes.md` is the GitHub Release body and describes the release event as a whole. It is produced from release-event text plus the per-binary notes referenced by each discovered binary manifest's `[release].notes_path`. It must list every binary in the release, each binary version, supported distribution channels, target platforms, high-level changes, install commands, verification/checksum guidance, and known issues. Per-binary notes such as `releases/mmux.md` carry binary-specific changes, CLI/API changes, install examples, target matrix, and compatibility notes.

Release notes are drafted when the release branch opens and finalized before the GitHub Release is created. The release skill should help the operator draft per-binary notes from each discovered manifest, then aggregate `releases/notes.md` from those note sources and a human-provided summary of user-visible release-event changes. The skill may use `git log` as supporting evidence, but it must not invent release claims from commit subjects alone. A human must approve the release notes before the `gh release create --notes-file releases/notes.md` step.

The final release branch ledger should record the GitHub Release notes URL and, if release notes are uploaded as assets, the checksum or URL for the final note files. Do not move the release tag to revise notes-only ledger metadata.

### npm

The npm registry hosts generated native packages under the `@motlie` org. Package templates and release scripts should live in the Motlie repo, but generated package directories do not need to be committed unless the PLAN chooses that workflow.

Native package example, parameterized by binary:

```json
{
  "name": "@motlie/<bin>-linux-x64-musl",
  "version": "0.1.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "libc": "musl",
  "files": ["bin/<bin>", "README.md", "LICENSE"],
  "bin": {
    "<bin>": "bin/<bin>"
  }
}
```

Worked `mmux` package example:

```json
{
  "name": "@motlie/mmux-linux-x64-musl",
  "version": "0.1.0",
  "os": ["linux"],
  "cpu": ["x64"],
  "libc": "musl",
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
  url "https://github.com/chungers/motlie/archive/refs/tags/2026-05-amber-aardvark.tar.gz"
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
motlie-mmux-v0.1.0-linux-x64-musl.tar.gz
motlie-mmux-v0.1.0-linux-arm64-musl.tar.gz
motlie-models-v0.1.0-linux-x64-gnu.tar.gz
motlie-models-v0.1.0-linux-x64-gnu-cuda-12.4.tar.gz
```

The executable inside the archive keeps the stable command name:

```text
bin/<bin>
```

## Linux libc Policy

Linux targets default to static musl when the binary and its native dependencies support it. For `mmux` v0.1, the required Linux targets are `linux-x64-musl` and `linux-arm64-musl`.

Every `linux-*-musl` target must record static-link evidence:

- `file <binary>` showing the produced executable type;
- `ldd <binary>` showing the binary is not dynamically linked, or equivalent output and exit status;
- `readelf -d <binary>` showing no shared runtime dependencies.

Static musl is the preferred default because it avoids distro-specific glibc floors, works in Alpine/musl environments, and simplifies host-wide SSH integration by reducing runtime dependency auditing.

Musl's allocator can be slower than glibc's allocator in allocation-heavy workloads. This is not expected to matter for TUI tools such as `mmux`; high-throughput binaries should benchmark the release artifact and may opt into a Rust allocator such as `mimalloc` or `jemalloc` if allocation behavior is material.

GNU/glibc targets are fallback or additional targets only when static musl is not feasible, or when the binary depends on glibc-linked runtimes such as CUDA. Every `linux-*-gnu` target must then record:

- `glibc_build_host_version`: the build host glibc reported by `ldd --version`;
- `glibc_min_version`: the actual binary GLIBC symbol floor reported by `objdump -T <binary> | grep GLIBC_ | sort -u`.

These gnu values are intentionally separate. The build host glibc version explains the build environment, while the binary GLIBC symbol floor is the compatibility requirement users actually hit at runtime. If gnu artifacts are enabled, build them in a pinned, known-old glibc environment or explicitly document the runtime floor.

## npm Package Naming

Generic package names:

```text
@motlie/<bin>-darwin-arm64
@motlie/<bin>-darwin-x64
@motlie/<bin>-linux-x64-musl
@motlie/<bin>-linux-arm64-musl
```

GNU/glibc package names are generated only when the binary manifest explicitly enables gnu targets:

```text
@motlie/<bin>-linux-x64-gnu
@motlie/<bin>-linux-arm64-gnu
```

Worked `mmux` package names:

```text
@motlie/mmux-darwin-arm64
@motlie/mmux-darwin-x64
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

Canonical installer templates should live on `main` in the Motlie source repository. Per-binary templates use `bins/<bin>/install-template.sh`; shared helpers can live under `releases/install-templates/_shared/` without storing release-event ledger state on `main`:

```text
bins/
  mmux/
    install-template.sh
  motlie-models/
    install-template.sh
releases/
  install-templates/
    _shared/
      detect-os.sh
      detect-arch.sh
      detect-libc.sh
      detect-cuda.sh
      fetch-release-asset.sh
      verify-checksum.sh
```

Each release branch copies the canonical template into branch-local `releases/install/install-<bin>.sh` during Phase 2.4 and may patch release-specific values such as tag, asset names, and checksums into that copy. Installer behavior fixes discovered on a release branch should usually be cherry-picked back to the canonical template on `main`.

The release asset naming rule is `install-<bin>.sh`; the listed files are examples.

The canonical hosted installer scripts should be uploaded to version-pinned GitHub Releases:

```text
https://github.com/chungers/motlie/releases/download/2026-05-amber-aardvark/install-mmux.sh
https://github.com/chungers/motlie/releases/download/2026-05-amber-aardvark/install-motlie-models.sh
```

GitHub Pages may provide latest convenience entrypoints:

```text
https://motlie.github.io/install/mmux.sh
https://motlie.github.io/install/motlie-models.sh
```

Production and automation docs should prefer release-pinned GitHub Release URLs. Pages URLs are convenience shortcuts and should redirect to, or fetch from, the current release-pinned installer.

The GitHub Pages update process is optional and must be explicit:

1. Publish the version-pinned installer asset to the GitHub Release first.
2. Open a separate PR to the configured Pages repository, for example `motlie/motlie.github.io`, updating `/install/<bin>.sh` to redirect to or fetch the release-pinned installer.
3. Verify the Pages URL downloads the intended release-pinned installer and that the installer checksum matches the GitHub Release asset.
4. Record the Pages URL, Pages repository commit, and verification evidence in the retained release-branch ledger.

If no Pages repository is configured for a binary release, do not advertise a Pages URL for production use.

In npm mode, installer scripts can select platform-specific native packages instead of direct archives. For a CUDA-capable binary, npm-mode selection could choose:

```text
@motlie/models-linux-x64-gnu-cuda-12-4
```

or, when CUDA is not required and static musl is feasible, fall back to:

```text
@motlie/models-linux-x64-musl
```

A non-CUDA gnu fallback is allowed only when static musl is not feasible and the manifest enables the gnu target with glibc-floor evidence.

### Release Upload Sequence

1. Create a release branch from `main`, for example `release/2026-05-amber-aardvark`.
2. Add branch-local release files under `releases/`: `manifest.toml`, `notes.md`, and one stable `releases/<bin>.toml` plus `releases/<bin>.md` pair for every binary in scope.
3. Push the release branch. The release branch is the coordination surface and release ledger; it is not a PR that will merge to `main`.
4. Create a master tracking issue that links the release branch and manifest files.
5. Create scoped sub-issues for platform/channel/gate work. Each sub-issue instructs the operator to open a sub-PR back to the release branch.
6. Land platform-specific sub-PRs into the release branch. Each sub-PR updates branch-local manifest status with staging evidence and closes its sub-issue.
7. Cherry-pick source or process fixes back to `main` through normal PRs when those fixes matter outside the release branch. Do not merge the full release branch to `main`.
8. After all required gates are complete or explicitly deferred, generate final aggregate release notes and ledger updates from manifests plus sub-issue/PR evidence.
9. Create the final release tag from the final release-branch commit after explicit human approval.
10. Build, sign, and package final artifacts from that final tag.
11. Upload canonical archives, checksums, installer scripts, workspace manifest, per-binary manifests, and release notes to the `chungers/motlie` GitHub Release.
12. Generate and publish native npm packages from the same final build outputs.
13. Update `motlie/homebrew-tap` with the new formula version and source tarball checksum.
14. Run install verification for npm, direct installer, and Homebrew from final installed paths.
15. Commit final ledger status back to the retained release branch or upload a final ledger manifest as a GitHub Release asset. Do not move the release tag to include ledger-only metadata.
16. Close the master tracking issue only after the GitHub Release is live, all required package/install gates are complete or deferred, and final ledger state is pushed.

Important GitHub constraint: a full GitHub Release is tag-centric, not PR-centric. The final release tag must point to the exact release-branch source commit used for final artifacts. Staging builds performed earlier on the release branch are useful evidence, but if the final tag commit differs from the staging commit, final artifacts must be rebuilt or revalidated from the final tag.

Because release tags are calver-codenames rather than semver, GitHub's automatic latest-release heuristic may not match operator intent. The release workflow should explicitly set the stable release with `gh release edit <release-name> --latest` when publishing.

### Publishing Credentials

The DESIGN should prefer trusted publishing or short-lived CI credentials where supported. If manual tokens are required for the first release, the PLAN must isolate them to release workflows and avoid storing them in the source tree.

## Release Manifest

Release manifests are branch-local. A release branch contains a common `releases/` directory, but that directory is not merged to `main`:

```text
releases/
  manifest.toml
  notes.md
  mmux.toml
  mmux.md
  motlie-models.toml
  motlie-models.md
  install/
    install-mmux.sh
    install-motlie-models.sh
```

`releases/manifest.toml` is the workspace release ledger. It owns release-event identity, branch, tag, GitHub Release URL, workspace-scoped gates, global release defaults, and final completion summaries for each binary. It is not the source of truth for the binaries to build. The build system and release skill discover binaries by scanning `releases/*.toml`, excluding `releases/manifest.toml`, and parsing only manifests with `kind = "motlie.binary-release"`.

Per-binary manifest filenames are stable and unversioned: `releases/<bin>.toml`. The released version is stored in `[identity].version`, and the per-binary note source is stored in `[release].notes_path`. This prevents a release branch from accumulating multiple versioned manifest files for one binary and makes the release fan-out deterministic from the directory contents. Single-binary releases are the degenerate case: scanning finds one binary manifest.

The manifest set is both deterministic input and release ledger:

- Intent sections are immutable release inputs. Build, package, installer, npm, and Homebrew steps read these sections.
- Status sections are mutable staging evidence. Platform sub-PRs update them while targeting the release branch.
- Published sections are final ledger metadata recorded on the retained release branch and uploaded as GitHub Release assets.

The build system and release skill must not derive a name when a manifest provides an explicit value. This is necessary for cases such as `mmux`, where npm must install a native binary directly and must not use `mmux.sh`, `mmux.js`, or another runner.

Discovery rules:

- Read `releases/manifest.toml` for release-event identity, global defaults, workspace gates, and aggregate release-note path.
- Scan `releases/*.toml` in lexical order, excluding `releases/manifest.toml`.
- Parse only files with `kind = "motlie.binary-release"`.
- Require each binary manifest filename to be `releases/<identity.binary>.toml`.
- Require each binary manifest to contain `[identity].version` and `[release].notes_path`.
- Reject duplicate `[identity].binary` values in a release branch.
- Drive build, packaging, signing, npm, Homebrew, installer, and per-target gates from the discovered binary manifests.
- Generate the final `releases/notes.md` GitHub Release body by aggregating release-event text with each binary manifest's `[release].notes_path`.

Workspace manifest example:

```toml
schema_version = 1
kind = "motlie.release"
state = "planned"

[identity]
name = "2026-05-amber-aardvark"
release_month = "2026-05"
codename = "amber-aardvark"

[coordination]
source_repo = "chungers/motlie"
base_branch = "main"
release_branch = "release/2026-05-amber-aardvark"
sub_prs_allowed = true
main_merge_policy = "never-merge-release-branch"
main_fix_policy = "cherry-pick-source-fixes-only"

[tracking]
master_issue = ""
master_issue_state = "planned"
master_issue_policy = "close-after-github-release-live-and-final-ledger-pushed"
sub_issue_policy = "one-or-more-issues-per-platform-channel-or-binary-gate"

[release]
tag = "2026-05-amber-aardvark"
notes_path = "releases/notes.md"
github_release = ""
source_ref_policy = "final-artifacts-must-build-from-final-tag"

[discovery]
binary_manifest_glob = "releases/*.toml"
exclude = ["releases/manifest.toml"]
required_kind = "motlie.binary-release"
filename_policy = "releases/<identity.binary>.toml"

[defaults]
linux_default_libc = "musl"
linux_static_policy = "default-static-musl-when-feasible"

[[binary_completion]]
binary = "mmux"
version = "0.1.0"
manifest = "releases/mmux.toml"
notes_path = "releases/mmux.md"
state = "planned"
final_artifacts_complete = false
published = false

[[gate]]
id = "release-branch-created"
scope = "workspace"
state = "planned"
completed_at = ""
completed_by = ""
source_commit = ""
evidence = []

[[gate]]
id = "final-tag-created"
scope = "workspace"
state = "planned"
completed_at = ""
completed_by = ""
source_commit = ""
evidence = []

[[gate]]
id = "github-release-published"
scope = "workspace"
state = "planned"
completed_at = ""
completed_by = ""
source_commit = ""
evidence = []
```

Per-binary manifest example:

```toml
schema_version = 1
kind = "motlie.binary-release"
state = "planned"

[identity]
binary = "mmux"
version = "0.1.0"
release_name = "2026-05-amber-aardvark"

[release]
notes_path = "releases/mmux.md"

[build]
cargo_package = "motlie-mmux"
cargo_bin = "mmux"
profile = "release"
locked = true
cargo_lock_policy = "must-be-committed-and-unchanged-at-final-tag"

[toolchain]
rust_policy = "record-rustc-and-cargo-version-in-evidence"
darwin_cross = "cargo-zigbuild"
darwin_cross_policy = "v0-default-for-darwin-from-linux"
linux_default_libc = "musl"
linux_static_policy = "default-static-musl-when-feasible"
linux_musl_toolchain = "rustup + cargo build --target"
linux_musl_toolchain_policy = "default-for-pure-rust-static-musl"
linux_musl_zigbuild_policy = "use-cargo-zigbuild-when-c-deps-need-musl-aware-linker"
linux_gnu_policy = "fallback-for-glibc-or-cuda-runtime"
linux_gnu_glibc_floor_policy = "record-host-and-binary-glibc-floor-for-gnu-targets"
required_evidence_universal = ["rustc -Vv", "cargo -V"]
darwin_cross_required_evidence = ["cargo zigbuild -V", "zig version"]
linux_musl_required_evidence = ["file <binary>", "ldd <binary>", "readelf -d <binary>"]
linux_gnu_required_evidence = ["objdump -T <binary> | grep GLIBC_ | sort -u", "ldd --version"]

[signing]
identity = "adhoc"
macos_verify_installed_path = true

[install]
command = "mmux"
default_path = "/usr/local/bin/mmux"
force_command_safe = true

[archive]
enabled = true
linux_default_libc = "musl"
musl_enabled = true
gnu_enabled = false
asset_prefix = "motlie-mmux"
format = "tar.gz"
binary_path = "bin/mmux"
include = ["README.md", "LICENSE"]

[installer]
enabled = true
script_asset = "install-mmux.sh"
source_path = "releases/install/install-mmux.sh"
default_source = "archive"

[npm]
enabled = true
linux_default_libc = "musl"
musl_enabled = true
gnu_enabled = false
scope = "@motlie"
package_prefix = "mmux"
access = "public"
runner = "native-binary"
node_launcher = false
bin_command = "mmux"
bin_path = "bin/mmux"

[homebrew]
enabled = true
tap = "motlie/homebrew-tap"
formula = "mmux"
source = "source-tag"

[[target]]
id = "linux-x64-musl"
os = "linux"
arch = "x64"
libc = "musl"
linkage = "static"
rust_target = "x86_64-unknown-linux-musl"
archive_asset = "motlie-mmux-v0.1.0-linux-x64-musl.tar.gz"
npm_package = "@motlie/mmux-linux-x64-musl"
archive_binary_path = "bin/mmux"
npm_bin_path = "bin/mmux"

[target.status]
state = "planned"
completed_at = ""
completed_by = ""
source_commit = ""
evidence = []

[[gate]]
id = "linux-builds-staged"
target_id = "linux-x64-musl"
channel = "archive"
state = "planned"
completed_at = ""
completed_by = ""
source_commit = ""
deferred_reason = ""
evidence = []
```

Gate status values are intentionally simple: `planned`, `staged`, `complete`, `deferred`, or `failed`. `staged`, `complete`, `deferred`, and `failed` gates record at least `completed_at`, `completed_by`, `source_commit`, and structured `evidence`. These fields allow different agents or humans to pick up release work on different hosts without relying on conversational context.

Binary-manifest gate rows are keyed by `(id, target_id)`. Target-specific platform and package work uses the target id, for example `target_id = "linux-x64-musl"`. Global or rollup gates use `target_id = ""`; rollup rows set `rollup = true`. A rollup gate is complete only when all enabled target/channel gates it summarizes are complete or explicitly deferred.

Evidence entries use this minimal shape:

```toml
{ kind = "command-log", ref = "PR #123 comment", sha256 = "", note = "rustc -Vv recorded" }
```

Install verification is also manifest-tracked. A direct installer release should include `installer-validated` gates for each target that must run the release-pinned installer on a matching host. The rollup `installer-validated` gate is complete only after the target-specific installer checks complete or are explicitly deferred.

Disabled-channel gates are marked `deferred` at release-branch-open time with `deferred_reason = "channel disabled"`.

For future CUDA-capable binaries, a target may add `accelerator = "cuda-12-4"`. CPU/default targets omit the accelerator field entirely.

After publication, workspace and per-binary manifests are updated on the retained release branch:

```toml
state = "published"

[published]
tag = "2026-05-amber-aardvark"
github_release = "https://github.com/chungers/motlie/releases/tag/2026-05-amber-aardvark"
release_notes = "https://github.com/chungers/motlie/releases/tag/2026-05-amber-aardvark"

[[published.asset]]
target = "darwin-arm64"
name = "motlie-mmux-v0.1.0-darwin-arm64.tar.gz"
url = "https://github.com/chungers/motlie/releases/download/2026-05-amber-aardvark/motlie-mmux-v0.1.0-darwin-arm64.tar.gz"
sha256 = "<final-sha256>"
signed = true
```

Do not move the release tag just to include ledger-only metadata. Ledger finalization happens on the retained release branch and in uploaded manifest assets; `main` remains free of per-release ledger files.

## Core Motlie Work

The first release implementation should avoid adding runtime dependencies to core Motlie libraries. Some shared helpers may still be justified.

Candidate work:

- Common build metadata for `--version` and possibly `--version --json`.
- Runtime self-check helpers for external dependencies such as `tmux`, `ssh`, CUDA libraries, or model assets.
- Artifact and package naming helpers driven by the release manifest.
- A release manifest parser in an internal `xtask` or release helper crate, including `releases/*.toml` discovery, `kind = "motlie.binary-release"` filtering, `releases/<identity.binary>.toml` filename validation, duplicate binary rejection, and `[release].notes_path` validation.
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
- Which binaries cannot use static musl and need additional `linux-*-gnu` archive/npm targets with glibc-floor evidence?
- Should the first release enable optional GitHub Pages convenience installer URLs, or ship only version-pinned GitHub Release installer URLs until the Pages repository workflow is proven?
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
