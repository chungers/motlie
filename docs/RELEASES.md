# Release Playbook

## Changelog

- 2026-05-12, @gpt55-dgx: Initial user playbook for staged Motlie binary releases, Linux cross-compilation, macOS signing, npm publication, and Homebrew tap updates.
- 2026-05-12, @gpt55-dgx: Added step-by-step npm authentication guidance and linked the repo-local release skill.
- 2026-05-12, @gpt55-dgx: Generalized the release playbook around a user-specified binary target; `mmux` is now only the worked example.
- 2026-05-12, @gpt55-dgx: Clarified that CI workflows are future automation; the current release execution path is the manual process in `docs/PLAN_RELEASES.md`.
- 2026-05-12, @gpt55-dgx: Aligned the playbook to per-binary manifests under `releases/<bin>/<version>.toml` and a release coordination PR with sub-PR status updates.

## Scope

This playbook describes how to produce and publish Motlie native binary releases. It is operational guidance for the release operator and complements:

- `docs/DESIGN_RELEASES.md`
- `docs/PLAN_RELEASES.md`
- `.agents/skills/release/SKILL.md`
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

## Release Target Parameters

This playbook is generic. The release operator must identify the binary target before starting a release. `mmux` is the first worked example used to validate the workflow, not the only supported target.

Required target fields:

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
MANIFEST=releases/<bin>/<version>.toml
RELEASE_BRANCH=release/<bin>-v<version>
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
MANIFEST=releases/mmux/0.1.0.toml
RELEASE_BRANCH=release/mmux-v0.1.0
```

## Release Model

Use a release coordination PR instead of publishing directly from a local build. The coordination PR is a long-running branch from `main` that carries the version bump, per-release manifest, release notes, and status updates from platform-specific sub-PRs.

The release should move through these gates:

1. Release coordination branch and PR in `chungers/motlie`.
2. Per-release manifest under `releases/<bin>/<version>.toml`.
3. Platform or channel sub-PRs targeting the release branch.
4. Manifest status updates recording staging evidence.
5. Coordination PR merge to `main`.
6. Final source tag from `main`.
7. Final build, signing, and archive publication from the final tag.
8. npm package publication.
9. Homebrew tap PR and bottle publication.
10. Post-release ledger PR recording final URLs, checksums, and package links.

This keeps staging, final publication, and post-release audit metadata distinct. If a later gate fails, status remains inspectable in the manifest without moving the release tag.

## Release Coordination PR

Open a release coordination PR against `main` before building final binaries.

Recommended branch name:

```text
release/mmux-v0.1.0
```

The initial PR should include:

- Version updates in `Cargo.toml`.
- The release manifest, for example `releases/mmux/0.1.0.toml`.
- Release notes, for example `releases/mmux/0.1.0.md`.
- Source-side installer, npm, or Homebrew templates under `releases/<bin>/` if needed.
- Release docs and release skill updates.

Platform-specific work should land as sub-PRs targeting the release branch:

```text
release/mmux-v0.1.0-linux-x64 -> release/mmux-v0.1.0
release/mmux-v0.1.0-darwin-arm64 -> release/mmux-v0.1.0
release/mmux-v0.1.0-npm -> release/mmux-v0.1.0
```

Each sub-PR should update `releases/<bin>/<version>.toml` with status, source commit, timestamp, actor, and evidence links. Do not commit built binaries to git.

## Final Tag and GitHub Release

Create the release tag from `main` after the release coordination PR merges and all required gates are complete or explicitly deferred.

```sh
git switch main
git pull --ff-only
rg -n 'binary = "mmux"|version = "0.1.0"|tag = "v0.1.0"' releases/mmux/0.1.0.toml Cargo.toml
git tag v0.1.0
git push origin v0.1.0
```

Important: a GitHub Release is tag-centric, not PR-centric. The final tag must point to the exact source commit used for final artifacts. Staging builds from the release branch are useful evidence, but final artifacts must be rebuilt or revalidated from the final tag if the commit changed.

Create the GitHub Release from the committed release notes:

```sh
gh release create v0.1.0 \
  --repo chungers/motlie \
  --title "v0.1.0" \
  --notes-file releases/mmux/0.1.0.md
```

## Cross-Compile on Linux

The final build should run from the source tag and produce target artifacts listed in the manifest:

```text
linux-x64-gnu
linux-arm64-gnu
darwin-x64
darwin-arm64
```

The Linux build may use Rust cross-compilation tooling, Zig, or an osxcross-style macOS SDK toolchain for Darwin targets. The exact toolchain is an implementation detail of `docs/PLAN_RELEASES.md`, but the release contract is:

- all outputs are built from the release tag;
- target names match `releases/<bin>/<version>.toml`;
- artifact names use explicit manifest fields such as `archive_asset`;
- archive binary paths use explicit manifest fields such as `archive_binary_path`;
- Darwin binaries produced on Linux are not considered final until macOS signing verification passes.

Generic archive names are a convention only. If the manifest provides an explicit `archive_asset`, use the manifest value.

```text
motlie-${BIN}-v${VERSION}-linux-x64-gnu.tar.gz
motlie-${BIN}-v${VERSION}-linux-arm64-gnu.tar.gz
motlie-${BIN}-v${VERSION}-darwin-x64.tar.gz
motlie-${BIN}-v${VERSION}-darwin-arm64.tar.gz
```

Worked `mmux` archive names:

```text
motlie-mmux-v0.1.0-linux-x64-gnu.tar.gz
motlie-mmux-v0.1.0-linux-arm64-gnu.tar.gz
motlie-mmux-v0.1.0-darwin-x64.tar.gz
motlie-mmux-v0.1.0-darwin-arm64.tar.gz
```

Every archive should include:

```text
bin/${BIN}
README.md
LICENSE
```

Generate checksums for every final archive:

```sh
shasum -a 256 "motlie-${BIN}-v${VERSION}"-*.tar.gz > SHA256SUMS
```

Do not upload unsigned Darwin artifacts as final release assets. If candidate artifacts must be shared before final signing, use PR evidence or clearly marked staging assets, and rebuild or revalidate from the final tag before publishing.

## macOS Signing Gate

Run a separate manual or manually approved gate on macOS during staging and again for final release artifacts if the final tag differs from the staged source commit.

This step exists because Apple Silicon validates Mach-O code signatures at execution time. A binary that works from one path can fail from another path after copying, so the final installed-path behavior must be verified.

Minimum signing for this release path is ad-hoc signing:

```sh
codesign --force --sign - "bin/${BIN}"
codesign --verify --strict --verbose=2 "bin/${BIN}"
"bin/${BIN}" --version
```

Manual signing workflow for final artifacts:

1. Build or download Darwin artifacts produced from the final source tag.
2. Extract each archive.
3. Re-sign `bin/${BIN}`.
4. Verify the signature.
5. Execute `bin/${BIN} --version`.
6. Repack the archive with the signed binary.
7. Recompute checksums.
8. Upload signed Darwin archives and checksum file to the GitHub Release.

Apple Silicon final-path verification:

```sh
sudo install -m 755 "bin/${BIN}" "${INSTALL_PATH}"
sudo codesign --force --sign - "${INSTALL_PATH}"
codesign --verify --strict --verbose=2 "${INSTALL_PATH}"
"${INSTALL_PATH}" --version
```

This final-path verification is required for:

- direct installer release validation;
- Darwin npm package validation;
- Homebrew formula or bottle validation.

Developer ID signing and notarization are not required for the first release path. They are a later distribution-hardening step if public download/Gatekeeper UX requires it.

## Finalize GitHub Release

After final Linux builds and final macOS signing pass from the final source tag, create or update the GitHub Release so only final artifacts are presented as installable.

Final release assets should match the manifest's explicit `archive_asset` values. For `mmux`:

```text
motlie-${BIN}-v${VERSION}-linux-x64-gnu.tar.gz
motlie-${BIN}-v${VERSION}-linux-arm64-gnu.tar.gz
motlie-${BIN}-v${VERSION}-darwin-x64.tar.gz
motlie-${BIN}-v${VERSION}-darwin-arm64.tar.gz
SHA256SUMS
${INSTALLER}
```

The installer script should default to archive mode for host/SSH-safe binaries:

```sh
sh "${INSTALLER}" --source archive
```

It may also support npm mode:

```sh
sh "${INSTALLER}" --source npm
```

When the artifacts are final and checksums match, publish the GitHub Release as the stable release for the tag.

## npm Publication

npm packages are published after final GitHub Release artifacts exist and after macOS signing passes.

Publish native packages only. Do not publish a Node boot script as the binary runtime entrypoint.

Generic package names are a convention only. If the manifest provides explicit `npm_package` values, use the manifest values.

```text
${NPM_PREFIX}-linux-x64-gnu
${NPM_PREFIX}-linux-arm64-gnu
${NPM_PREFIX}-darwin-x64
${NPM_PREFIX}-darwin-arm64
```

Worked `mmux` package names:

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
bin/${BIN}
```

Each package should expose:

```json
{
  "bin": {
    "<bin>": "bin/<bin>"
  }
}
```

Publication workflow:

1. Generate package directories from the final signed/release artifacts.
2. Run `npm pack --dry-run` for each package.
3. Install each package locally or in CI.
4. Execute `${BIN} --version` from the npm-installed path.
5. Select the npm auth path.
6. Publish to the npm registry under `@motlie`.

### npm Authentication Steps

npm credentials are needed only for the publish operation. They are not needed for build, packaging, GitHub Release upload, installer verification, or Homebrew work.

No npm API key is needed for:

1. Linux cross-compilation.
2. macOS signing.
3. Archive creation.
4. Checksum generation.
5. GitHub Release asset upload.
6. npm package directory generation.
7. `npm pack --dry-run`.
8. Local package install verification from a generated `.tgz`.
9. Homebrew formula PR or bottle build.

Preferred auth path:

- Use npm trusted publishing from GitHub Actions when available.
- Configure trusted publishing for the package and the exact GitHub Actions workflow.
- Give the workflow `id-token: write`.
- Do not create or store `NPM_TOKEN` when trusted publishing is working.

Bootstrap fallback path:

1. Create a granular npm token scoped to the `@motlie` org/packages.
2. Store it as a GitHub Actions secret in `chungers/motlie`.
3. Name the secret `NPM_TOKEN`.
4. Use it only in the `npm publish` step.
5. Pass it as `NODE_AUTH_TOKEN`.
6. Do not write an authenticated `.npmrc` into the repository.
7. Revoke the token after trusted publishing is configured and verified.

Token name if needed:

```text
NPM_TOKEN
```

Token-backed publish step:

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: 24
    registry-url: https://registry.npmjs.org

- run: npm publish --access public
  working-directory: dist/npm/@motlie/<package-name>
  env:
    NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Trusted-publishing shape:

```yaml
permissions:
  contents: read
  id-token: write

steps:
  - uses: actions/setup-node@v4
    with:
      node-version: 24
      registry-url: https://registry.npmjs.org
  - run: npm publish --access public
    working-directory: dist/npm/@motlie/<package-name>
```

GitHub Actions publish jobs should run only for releases or tags, not normal PRs.

Recommended trigger:

```yaml
on:
  release:
    types: [published]
```

If future automation publishes from a release workflow, require a manual approval environment before running `npm publish`.

## Homebrew Publication

Homebrew is macOS-only in this release model and is published through:

```text
github.com/motlie/homebrew-tap
```

User install UX:

```sh
brew tap motlie/tap
brew install "${FORMULA}"
```

Homebrew publication is a separate PR to `motlie/homebrew-tap`.

The tap PR should update:

```text
Formula/<formula>.rb
```

Recommended formula source:

```ruby
url "https://github.com/chungers/motlie/archive/refs/tags/v0.1.0.tar.gz"
sha256 "<source-tarball-sha256>"
```

The formula should build from source and re-sign after install on macOS:

```ruby
def install
  system "cargo", "build", "--release", "--locked", "-p", "<cargo-package>"
  bin.install "target/release/<bin>"
  system "codesign", "--force", "--sign", "-", bin/"<bin>" if OS.mac?
end
```

The formula test must execute the installed binary:

```ruby
test do
  assert_match "<bin>", shell_output("#{bin}/<bin> --version")
end
```

Worked `mmux` formula values:

```text
FORMULA=mmux
CARGO_PACKAGE=motlie-mmux
BIN=mmux
```

Bottle workflow:

1. Open tap PR.
2. Run tap CI on macOS Apple Silicon and Intel.
3. Build bottles from the formula.
4. Run the formula test from the bottled install path.
5. Publish bottle metadata using the tap workflow.
6. Merge the tap PR after bottle verification.

The Homebrew tap may use the source tarball rather than the prebuilt GitHub Release archives. The bottle workflow is responsible for producing and publishing Homebrew-managed macOS bottles.

## Future GitHub Actions Structure

The manual v0 release process does not create these workflows. Use this section only when the release process is ready to be automated after the manual process has been validated.

Future automation should use multiple workflows or multiple jobs with explicit gates.

Recommended structure:

```text
.github/workflows/release-build.yml
.github/workflows/release-npm.yml
.github/workflows/release-installer.yml
```

Release build workflow:

- Trigger on the final source tag or stable release publication.
- Build Linux and Darwin target archives from Linux.
- Upload final artifacts/checksums to the GitHub Release.

macOS signing workflow:

- Trigger manually with `workflow_dispatch`, or through a protected environment.
- Download final Darwin artifacts built from the source tag.
- Sign, verify, repack, and replace Darwin assets.
- Verify `${INSTALL_PATH} --version` from a final copied path.

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

- [ ] Release target captured: `BIN`, `CARGO_PACKAGE`, `CARGO_BIN`, `VERSION`, enabled channels, and targets.
- [ ] Release coordination branch created from `main`.
- [ ] `releases/<bin>/<version>.toml` committed with release intent, explicit names, target matrix, and gates.
- [ ] `releases/<bin>/<version>.md` committed as release-note source.
- [ ] Coordination PR opened against `main`.
- [ ] Platform/channel sub-PRs merged into the release branch.
- [ ] Manifest status updated with staging evidence and source commits.
- [ ] Coordination PR merged to `main`.
- [ ] Final source tag pushed from `main`.
- [ ] Final Linux and Darwin artifacts built from the source tag.
- [ ] Final Darwin artifacts signed and verified on macOS.
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
- [ ] GitHub Release published as stable.
- [ ] Post-release ledger PR updates `releases/<bin>/<version>.toml` with final URLs, checksums, npm links, Homebrew tap commit, and install evidence.

## Rollback

If GitHub Release artifact validation fails:

- Leave the GitHub Release unpublished or replace invalid assets before announcing the release.
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
- Whether Darwin candidate artifacts should be shared as PR evidence, workflow artifacts, or clearly marked non-final staging assets before final signing.
- Whether npm publication should run from the source repo workflow or a separate release environment.
- Whether the Homebrew tap should be updated by a bot PR from `chungers/motlie` or manually after the source release is finalized.
- Whether future public macOS downloads require Developer ID signing and notarization beyond ad-hoc signing.
