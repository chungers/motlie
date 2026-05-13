---
name: release
description: Execute or plan Motlie binary releases across GitHub Releases, npm @motlie native packages, direct installers, and the motlie/homebrew-tap Homebrew tap. Use when asked to release, stage, publish, package, codesign, create release PRs, update release docs, or verify release artifacts.
---

# Motlie Release

Use this skill for Motlie binary release work. The release target is parameterized by the binary the human specifies. Do not assume `mmux` unless the human says `mmux` or the task is explicitly the first worked release validation.

Before changing anything:

- identify yourself as required by `AGENTS.md`
- check `git status --short --branch`
- confirm the target branch, release version, and release target
- do not publish npm packages, GitHub Releases, or Homebrew tap changes without explicit human approval
- do not commit unrelated files
- default to the manual v0 process in `docs/PLAN_RELEASES.md`; do not propose new CI jobs unless the human explicitly asks for automation
- use `releases/<bin>/<version>.toml` as the release source of truth and status ledger

Release target fields:

```text
BIN=<installed command name>
CARGO_PACKAGE=<cargo package name>
CARGO_BIN=<cargo binary name>
VERSION=<release version>
INSTALL_PATH=<default absolute install path, if any>
FORMULA=<Homebrew formula name, if enabled>
NPM_PREFIX=@motlie/<package-prefix>
INSTALLER=install-<bin>.sh
FORCE_COMMAND_SAFE=true|false
MANIFEST=releases/<bin>/<version>.toml
RELEASE_BRANCH=release/<bin>-v<version>
```

Worked example:

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

Primary references:

- release playbook: `docs/RELEASES.md`
- design: `docs/DESIGN_RELEASES.md`
- plan: `docs/PLAN_RELEASES.md`
- issue: `https://github.com/chungers/motlie/issues/234`

Manual v0 release sequence:

1. Create a release coordination branch from `main`, for example `release/mmux-v0.1.0`.
2. Add or update `releases/<bin>/<version>.toml` and `releases/<bin>/<version>.md`.
3. Open the coordination PR from the release branch to `main`.
4. Land platform/channel sub-PRs into the release branch.
5. Update manifest status with staging evidence, source commits, timestamps, actors, and links.
6. Merge the coordination PR to `main` after all required gates are complete or explicitly deferred.
7. Create the final source tag from `main`.
8. Build, sign, package, and publish final artifacts from the final tag.
9. Validate release-pinned installers and update `installer-validated` gates.
10. Publish native npm packages and update `motlie/homebrew-tap`.
11. Open a post-release ledger PR that marks the manifest `state = "published"` and records final URLs/checksums/package links.

Release PR source files:

- `Cargo.toml`: bump `[workspace.package].version` and fix release metadata.
- `bins/<bin>/Cargo.toml`: verify package name, bin name, and description; most binaries should inherit the workspace version.
- `releases/<bin>/<version>.toml`: deterministic release intent and mutable status ledger.
- `releases/<bin>/<version>.md`: release notes used by `gh release create`.
- `releases/<bin>/install/*`: direct installer sources, only when installer distribution is in scope.
- `releases/<bin>/npm/*`: npm native package templates, only when npm distribution is in scope.
- `releases/<bin>/homebrew/*`: source-side formula template or notes only; the live formula PR belongs in `motlie/homebrew-tap`.
- `docs/*`: update when behavior or workflow changes.

To stage macOS signing from another host:

```sh
gh pr checkout <release-pr-number>
git switch release/<bin>-v<version>
git pull --ff-only
cargo build --release --locked --target <rust-target> -p <cargo-package> --bin <cargo-bin>
codesign --force --sign - target/<rust-target>/release/<bin>
codesign --verify --strict --verbose=2 target/<rust-target>/release/<bin>
target/<rust-target>/release/<bin> --version
sudo install -m 755 target/<rust-target>/release/<bin> <install-path>
sudo codesign --force --sign - <install-path>
codesign --verify --strict --verbose=2 <install-path>
<install-path> --version
```

After the coordination PR merges to `main`:

```sh
git switch main
git pull --ff-only
rg -n 'binary = "<bin>"|version = "<version>"|tag = "v<version>"' releases/<bin>/<version>.toml Cargo.toml
git tag v<VERSION>
git push origin v<VERSION>
gh release create v<VERSION> --repo chungers/motlie --title "v<VERSION>" --notes-file releases/<bin>/<version>.md
```

GitHub constraints:

- A full GitHub Release is tag-centric, not PR-centric.
- The final tag must point to the exact source commit used for final artifacts.
- Staging evidence from a release branch does not replace final build/signing from the final tag if the commit changed.
- Do not move the release tag to include post-release ledger metadata.

Manifest status rules:

- Gate state values are `planned`, `staged`, `complete`, `deferred`, or `failed`.
- `staged`, `complete`, `deferred`, and `failed` gates must record `completed_at`, `completed_by`, `source_commit`, and `evidence`.
- Per-target status is a struct at `[target.status]`; do not use a bare status string.
- Gate rows are keyed by `(id, target_id)`. Use `target_id = ""` only for global gates or explicit rollups. Rollup rows set `rollup = true` and summarize target-specific rows.
- Rollup gates are complete only when all enabled target/channel gates they summarize are complete or explicitly deferred.
- Installer validation gates use `id = "installer-validated"` and should be target-specific when the installer must run on matching host platforms.
- Channel-disabled gates are marked `deferred` at coordination-PR-open time with `deferred_reason`.
- Evidence entries use `{ kind, ref, sha256?, note? }`; include toolchain versions for build and signing gates.
- Status fields are evidence only; intent fields drive artifact names, package names, binary paths, and installer behavior.
- The v0 Darwin-from-Linux toolchain is `cargo-zigbuild`; record `rustc -Vv`, `cargo -V`, `cargo zigbuild -V`, and `zig version` in evidence.
- Use merge commits for the coordination PR; do not squash or rebase the release branch because the merge history preserves sub-PR evidence.

Operator prompt workflow:

1. Read `MANIFEST` before proposing the next action.
2. Identify the first incomplete gate, its required platform, and whether the current host can perform it.
3. If another host/operator is needed, prompt with the exact branch/PR to pull and the manifest fields to update.
4. If the action creates tags, GitHub Releases, npm publications, or Homebrew tap changes, ask for explicit approval.
5. After a gate is performed, update manifest status through a PR or sub-PR with `completed_at`, `completed_by`, `source_commit`, and `evidence`.
6. For handoffs, reply with current state, next gate, required host/platform, command group, files to update, and approval needed.

Read references only when needed:

- npm publishing or token handling: `references/npm-auth.md`
- macOS `codesign` or Darwin artifact validation: `references/macos-signing.md`
- Homebrew formula/tap/bottle work: `references/homebrew-tap.md`
- end-to-end checklist: `references/release-checklist.md`
- operator handoff prompts or next-step prompting: `references/operator-prompts.md`

Hard requirements:

- native npm packages expose the requested binary directly; do not add a Node boot script.
- if the manifest says `runner = "native-binary"` and `node_launcher = false`, do not create `<bin>.js` or `<bin>.sh` as the npm runtime entrypoint.
- direct installers for host/SSH-safe binaries default to archive mode, not npm mode.
- if `force_command_safe = false`, archive mode is not mandatory for host login safety, but native binaries and explicit runtime paths are still required.
- Darwin binaries must be signed and verified from their installed path.
- npm auth is needed only at `npm publish` time unless trusted publishing is configured.
- release artifacts, npm packages, installer scripts, and Homebrew formulae must all trace back to the same source tag.
- final artifact, npm package, and binary path names must come from the manifest when explicit fields are present.

Use this response shape for release status:

```text
Release state:
- source tag:
- GitHub Release:
- artifacts:
- macOS signing:
- npm:
- Homebrew:
- remaining gates:
```
