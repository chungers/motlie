---
name: release
description: Execute or plan Motlie binary releases across GitHub Releases, npm @motlie native packages, direct installers, and the motlie/homebrew-tap Homebrew tap. Use when asked to release, stage, publish, package, codesign, create release branches, update release docs, or verify release artifacts.
---

# Motlie Release

Use this skill for Motlie release work. The release event is identified by a calver-codename branch and may include one or more binaries. Do not assume `mmux` unless the human says `mmux` or the task is explicitly the first worked release validation.

Before changing anything:

- identify yourself as required by `AGENTS.md`
- check `git status --short --branch`
- confirm the release branch, release tag, and binaries in scope
- do not publish npm packages, GitHub Releases, or Homebrew tap changes without explicit human approval
- do not commit unrelated files
- default to the manual v0 process in `docs/PLAN_RELEASES.md`; do not propose new CI jobs unless the human explicitly asks for automation
- use branch-local `releases/manifest.toml` as the workspace release source of truth; read every referenced per-binary manifest before acting
- never merge a release branch to `main`; cherry-pick source, doc, skill, or tooling fixes back to `main` through separate PRs when needed

Release event fields:

```text
RELEASE_NAME=<YYYY-MM-adjective-codename>
RELEASE_BRANCH=release/<release-name>
RELEASE_TAG=<release-name>
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
```

Per-binary fields:

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
BINARY_MANIFEST=releases/<bin>-<version>.toml
BINARY_NOTES=releases/<bin>-<version>.md
```

Worked example:

```text
RELEASE_NAME=2026-05-amber-aardvark
RELEASE_BRANCH=release/2026-05-amber-aardvark
RELEASE_TAG=2026-05-amber-aardvark
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
BIN=mmux
CARGO_PACKAGE=motlie-mmux
CARGO_BIN=mmux
VERSION=0.1.0
INSTALL_PATH=/usr/local/bin/mmux
FORMULA=mmux
NPM_PREFIX=@motlie/mmux
INSTALLER=install-mmux.sh
FORCE_COMMAND_SAFE=true
BINARY_MANIFEST=releases/mmux-0.1.0.toml
BINARY_NOTES=releases/mmux-0.1.0.md
```

Primary references:

- release playbook: `docs/RELEASES.md`
- design: `docs/DESIGN_RELEASES.md`
- plan: `docs/PLAN_RELEASES.md`
- issue: `https://github.com/chungers/motlie/issues/234`

Manual v0 release sequence:

1. Create a release branch from `main`, for example `release/2026-05-amber-aardvark`.
2. Add or update branch-local `releases/manifest.toml`, `releases/notes.md`, and one per-binary manifest and notes file per released binary.
3. Push the release branch. Do not open a PR that merges it to `main`.
4. Land platform/channel sub-PRs into the release branch.
5. Update manifest status with staging evidence, source commits, timestamps, actors, and links.
6. Cherry-pick reusable source, doc, skill, or tooling fixes to `main` through separate PRs when needed.
7. Create the final source tag from the release branch after all required gates are complete or explicitly deferred.
8. Build, sign, package, and publish final artifacts from the final tag.
9. Validate release-pinned installers and update `installer-validated` gates.
10. Publish native npm packages and update `motlie/homebrew-tap`.
11. Push a final release-branch ledger commit that marks manifests `state = "published"` and records final URLs/checksums/package links.

Release branch source files:

- `Cargo.toml`: bump `[workspace.package].version` and fix release metadata.
- `bins/<bin>/Cargo.toml`: verify package name, bin name, and description; most binaries should inherit the workspace version.
- `releases/manifest.toml`: workspace release intent and mutable workspace ledger.
- `releases/notes.md`: GitHub Release notes source.
- `releases/<bin>-<version>.toml`: deterministic per-binary release intent and mutable binary ledger.
- `releases/<bin>-<version>.md`: per-binary notes included from workspace notes.
- `releases/install/*`: direct installer sources, only when installer distribution is in scope.
- `releases/npm/*`: npm native package templates, only when npm distribution is in scope.
- `releases/homebrew/*`: source-side formula template or notes only; the live formula PR belongs in `motlie/homebrew-tap`.
- `docs/*` and `.agents/skills/release/*`: update on `main` by cherry-picking through a normal PR when behavior or workflow changes.

To stage macOS signing from another host:

```sh
gh pr checkout <release-pr-number>
git switch release/<release-name>
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

After release-branch gates are complete:

```sh
git switch release/<release-name>
git pull --ff-only
rg -n 'name = "<release-name>"|tag = "<release-name>"|binary = "<bin>"|version = "<version>"' releases Cargo.toml
git tag <release-name>
git push origin <release-name>
gh release create <release-name> --repo chungers/motlie --title "<release-name>" --notes-file releases/notes.md
gh release edit <release-name> --repo chungers/motlie --latest
```

GitHub constraints:

- A full GitHub Release is tag-centric, not PR-centric.
- The final tag must point to the exact source commit used for final artifacts.
- Staging evidence from a release branch does not replace final build/signing from the final tag if the commit changed.
- Do not move the release tag to include post-release ledger metadata.
- Release branches never merge to `main`; cherry-pick fixes back instead.

Manifest status rules:

- Gate state values are `planned`, `staged`, `complete`, `deferred`, or `failed`.
- `staged`, `complete`, `deferred`, and `failed` gates must record `completed_at`, `completed_by`, `source_commit`, and `evidence`.
- Per-target status is a struct at `[target.status]`; do not use a bare status string.
- Gate rows are keyed by `(id, target_id)`. Use `target_id = ""` only for global gates or explicit rollups. Rollup rows set `rollup = true` and summarize target-specific rows.
- Rollup gates are complete only when all enabled target/channel gates they summarize are complete or explicitly deferred.
- Installer validation gates use `id = "installer-validated"` and should be target-specific when the installer must run on matching host platforms.
- Channel-disabled gates are marked `deferred` when the release branch opens with `deferred_reason`.
- Evidence entries use `{ kind, ref, sha256?, note? }`; include toolchain versions for build and signing gates.
- Status fields are evidence only; intent fields drive artifact names, package names, binary paths, and installer behavior.
- Record universal build evidence `rustc -Vv` and `cargo -V`.
- The v0 Darwin-from-Linux toolchain is `cargo-zigbuild`; record `cargo zigbuild -V` and `zig version` for Darwin cross-build evidence.
- Linux targets default to static musl when feasible. For pure-Rust `linux-*-musl`, use `rustup + cargo build --target`; use `cargo-zigbuild` only when C dependencies need a musl-aware linker. Record `file <binary>`, `ldd <binary>`, and `readelf -d <binary>` static-link evidence.
- Generate `linux-*-gnu` targets only when the manifest enables gnu fallback/CUDA targets. For those targets, record `ldd --version`, `objdump -T <binary> | grep GLIBC_ | sort -u`, `glibc_build_host_version`, and `glibc_min_version`.
- Use merge commits for sub-PRs into the release branch when possible; do not merge the release branch to `main`.

Operator prompt workflow:

1. Read `WORKSPACE_MANIFEST`, then every referenced `BINARY_MANIFEST`, before proposing the next action.
2. Identify the first incomplete workspace gate or `(binary, gate, target_id)`, its required platform, and whether the current host can perform it.
3. If another host/operator is needed, prompt with the exact branch/PR to pull and the manifest fields to update.
4. If the action creates tags, GitHub Releases, npm publications, or Homebrew tap changes, ask for explicit approval.
5. After a gate is performed, update manifest status on the release branch or through a sub-PR with `completed_at`, `completed_by`, `source_commit`, and `evidence`.
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
