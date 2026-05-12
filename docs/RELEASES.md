# Release Playbook

## Changelog

- 2026-05-12, @gpt55-dgx: Initial user playbook for staged Motlie binary releases, Linux cross-compilation, macOS signing, npm publication, and Homebrew tap updates.

## Scope

This playbook describes how to produce and publish Motlie native binary releases. It is operational guidance for the release operator and complements:

- `docs/DESIGN_RELEASES.md`
- `docs/PLAN_RELEASES.md`
- Issue #234: https://github.com/chungers/motlie/issues/234

The source repository remains:

```text
github.com/chungers/motlie
```

The package destinations are:

```text
GitHub Releases: github.com/chungers/motlie/releases
npm:             @motlie
Homebrew tap:    github.com/motlie/homebrew-tap
```

## Release Model

Use a staged release workflow instead of publishing directly from a local build.

The release should move through these gates:

1. Release preparation PR in `chungers/motlie`.
2. GitHub Release draft or prerelease for binary staging.
3. Linux-based cross-compilation for Linux and macOS target artifacts.
4. Separate macOS signing verification step.
5. Final GitHub Release artifact publication.
6. npm package publication.
7. Homebrew tap PR and bottle publication.
8. Post-publish install verification.

This keeps binary production, signing, package-manager upload, and install verification distinct. If a later gate fails, earlier artifacts remain inspectable and reproducible.

## Release Preparation PR

Open a release preparation PR against `main` before building binaries.

The PR should include only release metadata and docs needed for the release:

- Version updates.
- Changelog or release notes.
- Release manifest updates.
- New target matrix entries.
- npm package metadata template changes.
- installer script changes.
- Homebrew formula template changes if needed.

Do not publish npm packages or Homebrew formula updates from this PR. Its purpose is to make the source tree and release manifest reviewable.

Recommended branch name:

```text
@gpt55-dgx/release-v0.1.0
```

After review, merge the release preparation PR into `main`.

## Tag and Stage Release

Create the release tag from `main` after the release preparation PR merges.

```sh
git switch main
git pull --ff-only
git tag v0.1.0
git push origin v0.1.0
```

Create a GitHub Release in draft or prerelease state first. The staged release is where build artifacts, checksums, and installer scripts are uploaded before package-manager publication.

Recommended staged release state:

```text
draft: false
prerelease: true
```

Use prerelease instead of draft if downstream CI needs to download release assets for signing or package dry-runs.

## Cross-Compile on Linux

The primary build workflow should run on Linux and produce all initial target artifacts:

```text
linux-x64-gnu
linux-arm64-gnu
darwin-x64
darwin-arm64
```

The Linux workflow may use Rust cross-compilation tooling, Zig, or an osxcross-style macOS SDK toolchain for Darwin targets. The exact toolchain is an implementation detail of `docs/PLAN_RELEASES.md`, but the release contract is:

- all outputs are built from the release tag;
- target names match `docs/DESIGN_RELEASES.md`;
- artifact names follow the canonical naming grammar;
- the installed executable remains `mmux`;
- Darwin binaries produced on Linux are not considered final until macOS signing verification passes.

Example archive names:

```text
motlie-mmux-v0.1.0-linux-x64-gnu.tar.gz
motlie-mmux-v0.1.0-linux-arm64-gnu.tar.gz
motlie-mmux-v0.1.0-darwin-x64.tar.gz
motlie-mmux-v0.1.0-darwin-arm64.tar.gz
```

Every archive should include:

```text
bin/mmux
README.md
LICENSE
```

Generate checksums for every staged archive:

```sh
shasum -a 256 motlie-mmux-v0.1.0-*.tar.gz > SHA256SUMS
```

Upload the unsigned or pre-signing staged artifacts to the prerelease with names that make the state clear if needed:

```text
motlie-mmux-v0.1.0-darwin-arm64.tar.gz
motlie-mmux-v0.1.0-darwin-x64.tar.gz
SHA256SUMS
```

If unsigned Darwin artifacts are uploaded before macOS signing, mark the GitHub Release notes clearly:

```text
Darwin artifacts are staged for signing verification and are not final until the macOS signing gate completes.
```

## macOS Signing Gate

Run a separate manual or manually approved workflow on macOS after Linux cross-compilation completes.

This step exists because Apple Silicon validates Mach-O code signatures at execution time. A binary that works from one path can fail from another path after copying, so the final installed-path behavior must be verified.

Minimum signing for this release path is ad-hoc signing:

```sh
codesign --force --sign - bin/mmux
codesign --verify --strict --verbose=2 bin/mmux
bin/mmux --version
```

Manual signing workflow:

1. Download the staged Darwin archives from the GitHub prerelease.
2. Extract each archive.
3. Re-sign `bin/mmux`.
4. Verify the signature.
5. Execute `bin/mmux --version`.
6. Repack the archive with the signed binary.
7. Recompute checksums.
8. Replace the staged Darwin archives and checksum file on the GitHub Release.

Apple Silicon final-path verification:

```sh
sudo install -m 755 bin/mmux /usr/local/bin/mmux
sudo codesign --force --sign - /usr/local/bin/mmux
codesign --verify --strict --verbose=2 /usr/local/bin/mmux
/usr/local/bin/mmux --version
```

This final-path verification is required for:

- direct installer release validation;
- Darwin npm package validation;
- Homebrew formula or bottle validation.

Developer ID signing and notarization are not required for the first release path. They are a later distribution-hardening step if public download/Gatekeeper UX requires it.

## Finalize GitHub Release

After Linux builds and macOS signing pass, update the GitHub Release so only final artifacts are presented as installable.

Final release assets should include:

```text
motlie-mmux-v0.1.0-linux-x64-gnu.tar.gz
motlie-mmux-v0.1.0-linux-arm64-gnu.tar.gz
motlie-mmux-v0.1.0-darwin-x64.tar.gz
motlie-mmux-v0.1.0-darwin-arm64.tar.gz
SHA256SUMS
install-mmux.sh
```

The installer script should default to archive mode for `mmux`:

```sh
sh install-mmux.sh --source archive
```

It may also support npm mode:

```sh
sh install-mmux.sh --source npm
```

When the artifacts are final and checksums match, change the release from prerelease to stable.

## npm Publication

npm packages are published after final GitHub Release artifacts exist and after macOS signing passes.

Publish native packages only. Do not publish a Node boot script as the `mmux` runtime entrypoint.

Expected packages:

```text
@motlie/mmux-linux-x64-gnu
@motlie/mmux-linux-arm64-gnu
@motlie/mmux-darwin-x64
@motlie/mmux-darwin-arm64
```

Package contents:

```text
package.json
README.md
LICENSE
bin/mmux
```

Each package should expose:

```json
{
  "bin": {
    "mmux": "bin/mmux"
  }
}
```

Publication workflow:

1. Generate package directories from the final signed/release artifacts.
2. Run `npm pack --dry-run` for each package.
3. Install each package locally or in CI.
4. Execute `mmux --version` from the npm-installed path.
5. Publish to the npm registry under `@motlie`.

Preferred authentication:

- Use npm trusted publishing from GitHub Actions when available.
- If bootstrap requires a token, use a granular npm token scoped to the `@motlie` packages and store it as a GitHub Actions secret.
- Do not commit npm tokens.

Token name if needed:

```text
NPM_TOKEN
```

GitHub Actions publish jobs should run only for releases or tags, not normal PRs.

Recommended trigger:

```yaml
on:
  release:
    types: [published]
```

If publishing from a prerelease staging workflow, require a manual approval environment before running `npm publish`.

## Homebrew Publication

Homebrew is macOS-only in this release model and is published through:

```text
github.com/motlie/homebrew-tap
```

User install UX:

```sh
brew tap motlie/tap
brew install mmux
```

Homebrew publication is a separate PR to `motlie/homebrew-tap`.

The tap PR should update:

```text
Formula/mmux.rb
```

Recommended formula source:

```ruby
url "https://github.com/chungers/motlie/archive/refs/tags/v0.1.0.tar.gz"
sha256 "<source-tarball-sha256>"
```

The formula should build from source and re-sign after install on macOS:

```ruby
def install
  system "cargo", "build", "--release", "--locked", "-p", "motlie-mmux"
  bin.install "target/release/mmux"
  system "codesign", "--force", "--sign", "-", bin/"mmux" if OS.mac?
end
```

The formula test must execute the installed binary:

```ruby
test do
  assert_match "mmux", shell_output("#{bin}/mmux --version")
end
```

Bottle workflow:

1. Open tap PR.
2. Run tap CI on macOS Apple Silicon and Intel.
3. Build bottles from the formula.
4. Run the formula test from the bottled install path.
5. Publish bottle metadata using the tap workflow.
6. Merge the tap PR after bottle verification.

The Homebrew tap may use the source tarball rather than the prebuilt GitHub Release archives. The bottle workflow is responsible for producing and publishing Homebrew-managed macOS bottles.

## GitHub Actions Structure

Use multiple workflows or multiple jobs with explicit gates.

Recommended structure:

```text
.github/workflows/release-build.yml
.github/workflows/release-npm.yml
.github/workflows/release-installer.yml
```

Release build workflow:

- Trigger on tag or release prerelease.
- Build Linux and Darwin target archives from Linux.
- Upload staged artifacts/checksums to the GitHub Release.

macOS signing workflow:

- Trigger manually with `workflow_dispatch`, or through a protected environment.
- Download staged Darwin artifacts.
- Sign, verify, repack, and replace Darwin assets.
- Verify `/usr/local/bin/mmux --version` from a final copied path.

npm publication workflow:

- Trigger on stable GitHub Release publication.
- Download final release artifacts.
- Generate npm packages.
- Dry-run package contents.
- Publish to `@motlie`.

Homebrew workflow:

- Runs in `motlie/homebrew-tap`, not in `chungers/motlie`.
- Triggered by a PR, `workflow_dispatch`, or repository dispatch from the source release workflow.
- Builds formula and bottles on macOS runners.

## Manual Release Checklist

- [ ] Release preparation PR merged to `main`.
- [ ] `main` is clean and current.
- [ ] Release tag pushed.
- [ ] GitHub prerelease created.
- [ ] Linux cross-compilation produced Linux and Darwin archives.
- [ ] Checksums generated.
- [ ] Staged artifacts uploaded to GitHub prerelease.
- [ ] macOS signing workflow completed.
- [ ] Darwin archives replaced with signed artifacts.
- [ ] Final checksums uploaded.
- [ ] Installer script uploaded.
- [ ] Direct installer verified from release-pinned URL.
- [ ] npm packages generated from final artifacts.
- [ ] `npm pack --dry-run` reviewed for each package.
- [ ] npm packages published to `@motlie`.
- [ ] npm installs verified for Linux and macOS packages.
- [ ] Homebrew tap PR opened against `motlie/homebrew-tap`.
- [ ] Homebrew formula builds from source tag.
- [ ] Homebrew formula re-signs installed binary on macOS.
- [ ] Homebrew bottle tests pass from installed path.
- [ ] Homebrew tap PR merged.
- [ ] GitHub Release marked stable.

## Rollback

If GitHub Release artifact validation fails:

- Keep the release as prerelease.
- Replace failed artifacts.
- Regenerate checksums.
- Re-run signing and install verification.

If npm publication fails before all packages are published:

- Stop publication.
- Do not publish a convenience/meta package.
- Fix and publish the missing native packages at the same version if npm permits.
- If a broken package version is already public, publish a new patch version.

If Homebrew publication fails:

- Leave the tap PR open.
- Fix the formula or bottle workflow in the tap PR.
- Do not change the source GitHub Release unless the source artifact is invalid.

## Open Decisions

- Which Linux cross-compilation toolchain should become the blessed release path for Darwin targets.
- Whether Darwin artifacts should be uploaded before signing as prerelease assets or kept only as workflow artifacts until signing completes.
- Whether npm publication should run from the source repo workflow or a separate release environment.
- Whether the Homebrew tap should be updated by a bot PR from `chungers/motlie` or manually after the source release is finalized.
- Whether future public macOS downloads require Developer ID signing and notarization beyond ad-hoc signing.
