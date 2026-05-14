# Release Playbook

## Changelog

- 2026-05-12, @gpt55-dgx: Initial user playbook for staged Motlie binary releases, Linux cross-compilation, macOS signing, npm publication, and Homebrew tap updates.
- 2026-05-12, @gpt55-dgx: Added step-by-step npm authentication guidance and linked the repo-local release skill.
- 2026-05-12, @gpt55-dgx: Generalized the release playbook around a user-specified binary target; `mmux` is now only the worked example.
- 2026-05-12, @gpt55-dgx: Clarified that CI workflows are future automation; the current release execution path is the manual process in `docs/PLAN_RELEASES.md`.
- 2026-05-12, @gpt55-dgx: Aligned the playbook to per-binary manifests under `releases/<bin>/<version>.toml` and a release coordination PR with sub-PR status updates.
- 2026-05-12, @gpt55-dgx: Added operator handoff workflow showing how the release skill prompts humans and updates manifest state at each release gate.
- 2026-05-13, @gpt55-dgx: Tightened manifest status schema, target-specific gates, evidence requirements, merge strategy, disabled-channel handling, and v0 Darwin cross-build toolchain guidance.
- 2026-05-13, @gpt55-dgx: Added manifest-tracked installer validation, detached-tag build guidance, GitHub Pages installer update rules, and release rollback semantics.
- 2026-05-13, @gpt55-dgx: Made static musl the default Linux artifact policy when feasible, with glibc-floor evidence only for gnu fallback/CUDA targets.
- 2026-05-14, @gpt55-dgx: Split evidence requirements by target category and clarified the default Linux musl build toolchain.

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

For `FORCE_COMMAND_SAFE=true`, the direct installer must default to archive mode and the runtime path must execute the native binary directly. For `FORCE_COMMAND_SAFE=false`, npm-mode install may be acceptable for non-login use cases, but native binaries and explicit runtime paths are still required.

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

## Operator Handoff and Skill Prompts

Release work can be performed by different humans or agents on different hosts. The release manifest is the handoff document. The release skill is responsible for reading the manifest, identifying the next gate, prompting the human for the action appropriate to that gate, and recording evidence through a PR update.

At the start of every release turn, the skill should:

1. Identify itself and check `git status --short --branch`.
2. Read `releases/<bin>/<version>.toml`.
3. Summarize the current release state, incomplete gates, and the branch or PR the operator should work on.
4. Ask for explicit approval before publishing, tagging, modifying package registries, or changing Homebrew tap state.
5. Update only manifest status/evidence for staging work; final published URLs and checksums belong in the post-release ledger PR.

Operator prompts should be concrete. The prompt should tell the human what host/platform is needed, what branch to pull, what command group will run, and what manifest gate will be updated.

Gate rows are keyed by `(id, target_id)`. Use `target_id = ""` only for global gates or explicit rollups. A rollup row sets `rollup = true` and is complete only when every enabled target-specific row for that gate is `complete` or explicitly `deferred`. Platform/channel work should update target-specific rows first so concurrent operators do not collide on one coarse gate. Gate evidence entries use this shape:

```toml
evidence = [
  { kind = "command-log", ref = "PR #123 comment", sha256 = "", note = "rustc -Vv and cargo -V output recorded" },
]
```

For `staged`, `complete`, `deferred`, and `failed`, the gate or target status must record `completed_at`, `completed_by`, `source_commit`, and `evidence`. Disabled-channel gates are marked `deferred` when the coordination PR opens, with `deferred_reason = "channel disabled"`.

| Release gate | Operator surface | Skill prompt and action |
| --- | --- | --- |
| Intake | `main` | Confirm `BIN`, `VERSION`, `MANIFEST`, `RELEASE_BRANCH`, enabled channels, and platform targets. If missing, prompt for the missing field before editing files. |
| Coordination PR | `release/<bin>-v<version> -> main` | Prompt to create the release branch, add `releases/<bin>/<version>.toml`, add release notes, and open the coordination PR. Update the `release-pr-opened` gate with PR URL and source commit. |
| Linux staging | sub-PR to release branch | Prompt the Linux operator to pull the release branch, build the scoped `target_id`, package or validate artifact names from the manifest, and update the corresponding target status plus `(id, target_id)` gate with commit, checksum, toolchain, and evidence. |
| macOS staging | sub-PR to release branch | Prompt the macOS operator to pull the release branch, build with manifest `rust_target`, run build-path and installed-path `codesign` checks, and update the Darwin signing gate with timestamp, actor, source commit, signing identity, and evidence. |
| npm staging | sub-PR to release branch | Prompt the operator to generate one native package candidate per `target_id` using manifest `npm_package`, `bin_path`, and `node_launcher = false`; run `npm pack --dry-run`; update target-specific npm gate status only. |
| Homebrew staging | tap PR or source-side template PR | Prompt the operator to prepare the tap PR shape and record tap PR evidence. Do not merge live tap changes until final source tag exists. |
| Coordination merge | coordination PR | Prompt the human reviewer to confirm all required gates are `complete` or explicitly `deferred`, then merge to `main` with a merge commit. Warn that final artifacts must trace to the final tag. |
| Final tag | `main` | Prompt for explicit approval to create and push `v<VERSION>`. Verify the manifest tag and workspace version before tagging. |
| GitHub Release | final tag | Prompt for explicit approval to create the GitHub Release and upload final assets. Use manifest asset names and release notes. |
| Installer validation | final GitHub Release assets | Prompt the operator to run the release-pinned installer on each target platform, execute `<bin> --version` from the installed path, and update the target-specific `installer-validated` gate. |
| npm publish | final artifacts | Prompt for explicit approval and auth mode. Publish only after final artifacts exist and package dry-runs/install tests pass. |
| Homebrew publish | `motlie/homebrew-tap` | Prompt for tap PR merge or bottle publication only after the final source tag exists and formula tests pass. |
| Post-release ledger | short PR to `main` | Prompt to update `state = "published"` and record final URLs, checksums, npm links, Homebrew tap commit, and install evidence. Never move the release tag for ledger-only metadata. |

The skill reference `.agents/skills/release/references/operator-prompts.md` contains the step-specific prompt templates. Operators should prefer those prompts over reconstructing the process from memory.

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

Use merge commits for the coordination PR. Do not squash or rebase the release branch, because preserving the sub-PR merge history makes manifest evidence and platform handoffs easier to audit.

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

Build final artifacts from a detached checkout of the tag so the source tree cannot accidentally include release-branch or working-tree changes:

```sh
git switch --detach v0.1.0
git status --short --branch
```

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
linux-x64-musl
linux-arm64-musl
darwin-x64
darwin-arm64
```

The Linux build uses static musl targets by default when feasible and uses `cargo-zigbuild` as the default v0 Darwin cross-build path. A release may document an approved exception in manifest evidence, but the normal release contract is:

- all outputs are built from the release tag;
- `Cargo.lock` is committed and unchanged at the final tag;
- target names match `releases/<bin>/<version>.toml`;
- artifact names use explicit manifest fields such as `archive_asset`;
- archive binary paths use explicit manifest fields such as `archive_binary_path`;
- `linux-*-musl` targets record static-link evidence;
- `linux-*-gnu` fallback targets record both build-host glibc and binary GLIBC symbol floor;
- Darwin binaries produced on Linux are not considered final until macOS signing verification passes.

Universal build evidence must record `rustc -Vv` and `cargo -V`. For Darwin cross builds, also record `cargo zigbuild -V` and `zig version`.

For pure-Rust `linux-*-musl` targets, the v0 default toolchain is `rustup + cargo build --target`. Use `cargo-zigbuild` for musl only when C dependencies need a musl-aware linker. Musl evidence must record `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` to show the binary has no shared runtime dependencies. If `rust-toolchain.toml` is not present for the release tag, the exact Rust toolchain identity in evidence is mandatory.

If a release enables `linux-*-gnu` targets because static musl is not feasible or because a glibc-linked runtime such as CUDA is required, evidence must also record `ldd --version` for the build host and `objdump -T <binary> | grep GLIBC_ | sort -u` for the actual binary GLIBC requirement. Populate `glibc_build_host_version` from `ldd --version` and `glibc_min_version` from the highest GLIBC symbol required by the built binary.

For `mmux` v0.1, leave `gnu_enabled = false` and make `musl_enabled = true`.

Generic archive names are a convention only. If the manifest provides an explicit `archive_asset`, use the manifest value.

```text
motlie-${BIN}-v${VERSION}-linux-x64-musl.tar.gz
motlie-${BIN}-v${VERSION}-linux-arm64-musl.tar.gz
motlie-${BIN}-v${VERSION}-darwin-x64.tar.gz
motlie-${BIN}-v${VERSION}-darwin-arm64.tar.gz
```

Worked `mmux` archive names:

```text
motlie-mmux-v0.1.0-linux-x64-musl.tar.gz
motlie-mmux-v0.1.0-linux-arm64-musl.tar.gz
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
motlie-${BIN}-v${VERSION}-linux-x64-musl.tar.gz
motlie-${BIN}-v${VERSION}-linux-arm64-musl.tar.gz
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

After final assets are uploaded, validate the release-pinned installer URL on each supported target and update the corresponding `installer-validated` gate in `releases/<bin>/<version>.toml`:

```sh
curl -fsSLO https://github.com/chungers/motlie/releases/download/v0.1.0/install-mmux.sh
sh install-mmux.sh --source archive --prefix /usr/local
/usr/local/bin/mmux --version
```

GitHub Pages installer URLs are optional convenience entrypoints. If enabled for a release, update them only after the version-pinned GitHub Release installer exists:

1. Open a PR to the configured Pages repository, for example `motlie/motlie.github.io`.
2. Update `/install/<bin>.sh` to redirect to or fetch the current release-pinned GitHub Release installer.
3. Verify the Pages URL downloads the intended release-pinned script and that checksums match.
4. Record the Pages URL, Pages commit, and verification evidence in the post-release ledger PR.

If no Pages repository is configured, skip this channel and do not advertise the Pages URL as a production install path.

When the artifacts are final and checksums match, publish the GitHub Release as the stable release for the tag.

## npm Publication

npm packages are published after final GitHub Release artifacts exist and after macOS signing passes.

Publish native packages only. Do not publish a Node boot script as the binary runtime entrypoint.

Generic package names are a convention only. If the manifest provides explicit `npm_package` values, use the manifest values.

```text
${NPM_PREFIX}-linux-x64-musl
${NPM_PREFIX}-linux-arm64-musl
${NPM_PREFIX}-darwin-x64
${NPM_PREFIX}-darwin-arm64
```

Worked `mmux` package names:

```text
@motlie/mmux-linux-x64-musl
@motlie/mmux-linux-arm64-musl
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
- Keep npm provenance enabled; trusted publishing can attach registry provenance without a long-lived npm token.
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
- [ ] `releases/<bin>/<version>.toml` committed with release intent, explicit names, target matrix, structured target status, and `(id, target_id)` gates.
- [ ] `releases/<bin>/<version>.md` committed as release-note source.
- [ ] Disabled-channel gates are absent or marked `deferred` with `deferred_reason = "channel disabled"`.
- [ ] Coordination PR opened against `main`.
- [ ] Platform/channel sub-PRs merged into the release branch.
- [ ] Manifest status updated with target id, channel, staging evidence, toolchain evidence, and source commits.
- [ ] `linux-*-musl` target evidence records `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` static-link evidence.
- [ ] Any enabled `linux-*-gnu` fallback target evidence records `glibc_build_host_version`, `glibc_min_version`, `ldd --version`, and `objdump -T` GLIBC symbols.
- [ ] `Cargo.lock` committed and unchanged at the final source tag.
- [ ] Coordination PR merged to `main` with a merge commit.
- [ ] Final source tag pushed from `main`.
- [ ] Final Linux and Darwin artifacts built from the source tag.
- [ ] Final Darwin artifacts signed and verified on macOS.
- [ ] Final checksums uploaded.
- [ ] Installer script uploaded.
- [ ] Direct installer verified from release-pinned URL and `installer-validated` gates updated.
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
- Replace failed assets with explicit operator action, for example delete/re-upload the asset or use `gh release upload --clobber` while the release is still unannounced.
- Regenerate checksums and re-upload checksum assets with the same replacement discipline.
- Re-run signing and install verification.
- If an announced asset is broken, prefer a new patch release instead of silently replacing an asset users may already have cached.

If npm publication fails before all packages are published:

- Stop publication.
- Do not publish a convenience/meta package.
- Fix and publish the missing native packages at the same version if npm permits.
- If a broken package version is already public, prefer `npm deprecate <package>@<version> "<reason>"` and publish a new patch version.
- Use `npm unpublish` only when npm policy allows it and the release owner explicitly approves; newly created packages have a 72-hour unpublish window only when policy criteria allow, unpublished package versions cannot be reused, and unpublish should not be the normal rollback plan. See npm's unpublish policy: https://docs.npmjs.com/policies/unpublish

If Homebrew publication fails:

- Leave the tap PR open.
- Fix the formula or bottle workflow in the tap PR.
- Do not change the source GitHub Release unless the source artifact is invalid.

## Open Decisions

- Whether Darwin candidate artifacts should be shared as PR evidence, workflow artifacts, or clearly marked non-final staging assets before final signing.
- Whether npm publication should run from the source repo workflow or a separate release environment.
- Whether the Homebrew tap should be updated by a bot PR from `chungers/motlie` or manually after the source release is finalized.
- Whether future public macOS downloads require Developer ID signing and notarization beyond ad-hoc signing.
