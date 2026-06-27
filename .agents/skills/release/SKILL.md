---
name: release
description: Execute or plan Motlie binary releases across GitHub Releases, npm @motlie native packages, direct installers, and the motlie/homebrew-tap Homebrew tap. Use when asked to release, stage, publish, package, codesign, suggest release codenames, create release branches, create release tracking issues, update release docs, or verify release artifacts.
---

# Motlie Release

Use this skill for Motlie release work. The release event is identified by a calver-codename branch and may include one or more binaries. Do not assume `mmux` unless the human says `mmux` or the task is explicitly the first worked release validation.

Before changing anything:

- identify yourself as required by `AGENTS.md`
- check `git status --short --branch`
- confirm the release branch, release tag, and binaries in scope
- for new releases, suggest 3-5 calver-codename candidates, check remote branch/tag conflicts, and ask the human to select the release name
- do not publish npm packages, GitHub Releases, or Homebrew tap changes without explicit human approval
- do not commit unrelated files
- default to the manual v0 process in `docs/PLAN_RELEASES.md`; do not propose new CI jobs unless the human explicitly asks for automation
- use branch-local `releases/manifest.toml` as the workspace release ledger; scan `releases/*.toml` for stable per-binary manifests before acting
- never merge a release branch to `main`; cherry-pick source, doc, skill, or tooling fixes back to `main` through separate PRs when needed

Release event fields:

```text
RELEASE_NAME=<YYYY-MM-codename>
RELEASE_BRANCH=release/<release-name>
RELEASE_TAG=<release-name>
WORKSPACE_MANIFEST=releases/manifest.toml
WORKSPACE_NOTES=releases/notes.md
```

Per-binary manifests are discovered by scanning `releases/*.toml`, excluding `releases/manifest.toml`, and parsing files with `kind = "motlie.binary-release"`. Version belongs in `[identity].version`, not the filename.

Per-binary manifest fields:

```text
BINARY_MANIFEST=releases/<bin>.toml
BINARY_NOTES=releases/<bin>.md
[identity].binary=<installed command name>
[identity].version=<release version>
[build].cargo_package=<cargo package name>
[build].cargo_bin=<cargo binary name>
[install].default_path=<default absolute install path, if any>
[install].force_command_safe=true|false
[release].notes_path=releases/<bin>.md
```

Worked example:

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

Primary references:

- release playbook: `docs/RELEASES.md`
- design: `docs/DESIGN_RELEASES.md`
- plan: `docs/PLAN_RELEASES.md`
- issue: `https://github.com/chungers/motlie/issues/234`

Manual v0 release sequence:

1. Suggest release codenames, check conflicts, and get human confirmation for release name and binaries.
2. Create a release branch from `main`, for example `release/2026-05-amber-aardvark`.
3. Add or update branch-local `releases/manifest.toml`, `releases/notes.md`, and one stable `releases/<bin>.toml` plus referenced notes file per released binary.
4. Push the release branch. Do not open a PR that merges it to `main`.
5. Create a master tracking issue and record it in the workspace manifest tracking metadata.
6. Create scoped sub-issues for platform/channel/gate work; each should instruct the operator to open a sub-PR back to the release branch.
7. Land platform/channel sub-PRs into the release branch; each sub-PR updates manifest status and closes its sub-issue. Note: merging into the **release branch does NOT auto-close** a sub-PR's `Closes #N` issue (GitHub only auto-closes on the default branch), so close sub-issues explicitly with `gh issue close` when their gate completes.
8. Cherry-pick reusable source, doc, skill, or tooling fixes to `main` through separate PRs when needed. Use `git cherry-pick -x` so each commit carries the `(cherry picked from commit <release-branch-sha>)` lineage trailer; see the "Cherry-pick fixes from a release branch to `main`" section below for the full procedure.
9. Generate final notes/ledger state from manifests plus issue/PR evidence after required gates complete or defer.
10. Create the final source tag from the release branch after explicit approval.
11. Build, sign, package, and publish final artifacts from the final tag.
12. Validate release-pinned installers and update `installer-validated` gates. The installer downloads its asset from the GitHub Release, so this step is **gated on step 11 (publish) and cannot run before the Release exists** — never attempt installer-validation pre-publish. Run the published `curl|sh` installer on a host matching each target arch (a user-writable `PREFIX` avoids sudo); record host, command, installer-output, install-path, runtime `--version`, and (Darwin) `codesign --verify` evidence.
13. Publish native npm packages and update `motlie/homebrew-tap`.
14. Push a final release-branch ledger commit that marks manifests `state = "published"` and records final URLs/checksums/package links.
15. Close the master issue only after the GitHub Release is live and final ledger state is pushed.

Release branch source files:

- `Cargo.toml`: bump `[workspace.package].version` and fix release metadata.
- `bins/<bin>/Cargo.toml`: verify package name, bin name, and description; most binaries should inherit the workspace version.
- `releases/manifest.toml`: workspace release identity, global defaults, discovery policy, workspace gates, and final completion ledger.
- `releases/notes.md`: human-approved GitHub Release notes source.
- `releases/<bin>.toml`: deterministic per-binary release intent, version, and mutable binary ledger.
- `releases/<bin>.md`: per-binary notes referenced by the binary manifest and aggregated into workspace notes.
- `releases/install/*`: direct installer sources copied from canonical templates such as `bins/<bin>/install-template.sh`, only when installer distribution is in scope.
- `releases/npm/*`: npm native package templates, only when npm distribution is in scope.
- `releases/homebrew/*`: source-side formula template or notes only; the live formula PR belongs in `motlie/homebrew-tap`.
- `docs/*` and `.agents/skills/release/*`: update on `main` by cherry-picking through a normal PR when behavior or workflow changes.

Cherry-pick fixes from a release branch to `main`:

The release branch is the authoritative surface during a release event. Fixes (source code, docs, skill updates, tooling) discovered while working a release **always land on the release branch first**, via a sub-PR scoped to that fix. They land on `main` afterward by cherry-picking the resulting release-branch commit into a separate PR targeting `main`. **The release branch is never merged into `main`** — only specific commits are cherry-picked.

Sequence:

```
fix needed during release event
  └─ sub-PR against release/<release-name>  ─ merge ─►  release branch (fix is now in the release)
                                                          └─ wait until merged on release branch
                                                                └─ later: branch off main, cherry-pick -x from
                                                                   the release-branch commit, PR to main
```

Required procedure for the cherry-pick step (after the fix has merged into the release branch):

```sh
# 1. Branch off main, not off the release branch — keeps the PR diff scoped to just the cherry-pick(s).
git switch main
git pull --ff-only
git switch -c <your-identity>/<short-scope>          # e.g. @opus47-rel-eng/mmux-version-cli

# 2. Cherry-pick with -x so each cherry-pick commit message carries the lineage trailer
#    "(cherry picked from commit <release-branch-sha>)".
#    The <release-branch-sha> is the post-merge commit on the release branch (the sub-PR's commit, or
#    its post-merge form). Verify with `git log release/<release-name>` first.
git cherry-pick -x <release-branch-sha> [<another-sha>...]

# 3. Verify locally (build + relevant tests still pass on main + cherry-pick).
cargo build --release --locked -p <cargo-package> --bin <cargo-bin>
cargo test  --release --locked -p <cargo-package> [<test-filter>]

# 4. Push and open a PR to main.
git push -u origin <branch>
gh pr create --base main --head <branch> \
  --title '<scope summary>' --body '<links to release branch + original commits + master release issue>'
```

`-x` is required, not optional. Without it the cherry-pick commit message omits the `(cherry picked from commit ...)` trailer and `git log` on `main` can no longer trace the commit back to the release branch where it was first written. The trailer is what makes a cherry-pick PR provenanceably "from the release branch" even though the branch base is `main`.

PR body conventions for cherry-pick PRs:

- Title is the scope summary (e.g., `mmux: add --version flag and Help modal version line`), not "cherry-pick of X".
- Body links to the release branch (`release/<release-name>`), the original release-branch commits, and the master release issue that flagged the deferred cherry-pick.
- Body lists each cherry-picked commit with `cherry-pick SHA | original SHA | scope` so reviewers can audit.
- Multiple related commits may be bundled in one PR if they're a logical unit (e.g., a CLI fix + its matching test).
- Branch base is `main`. Branch ancestor is `main` (not the release branch tip), so the PR diff stays scoped to the cherry-pick.

Timing: cherry-picks can be opened at any point after the fix has merged into the release branch. They don't have to wait for the release event to close, and they don't all have to land in one batch. Land them when ready, paced by review capacity.

To stage macOS signing from another host, use a scoped sub-issue and sub-PR targeting the release branch. Read `references/macos-signing.md` and fill commands from the current per-binary manifest.

```sh
git switch release/<release-name>
git pull --ff-only
# create a short branch from the release branch, update the relevant manifest gate,
# then open a PR back to release/<release-name> that closes the scoped sub-issue.
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
- Rust builds are not byte-reproducible (embedded build paths / `BuildID`), so the final from-tag artifact and binary sha256 will differ from staging **even when the source is unchanged** (e.g. only docs changed since staging). Treat the final from-tag shas as authoritative, publish those, and record them in the ledger; the staging shas are superseded — do not block on staging-vs-final sha mismatch.
- Do not move the release tag to include post-release ledger metadata.
- Release branches never merge to `main`; cherry-pick fixes back instead.
- Concurrent release branches are allowed. Pull relevant fixes from `main` into another release branch with a normal merge from `main`.

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

Package build rules:

- Build archives and npm package directories from final artifacts produced from `RELEASE_TAG`; staging package directories are evidence only.
- When packaging **Darwin** archives on macOS, set `COPYFILE_DISABLE=1` and strip xattrs (`tar --no-xattrs`, or bsdtar `--no-mac-metadata`) so the archive contains only `bin/<binary>`. Plain macOS `tar` injects AppleDouble `._*` resource-fork files and `com.apple.provenance` xattrs that would otherwise ship in the public release. Verify with `tar -tzf <archive>` (expect only `bin/<binary>`, no `._*`) before publishing.
- Native npm package directories are generated from per-target `npm_package`, `npm_bin_path`, `bin_command`, and `node_launcher = false`; run `npm pack --dry-run`, install the generated `.tgz`, and execute the installed binary before publishing.
- Homebrew release work happens in `motlie/homebrew-tap`; formulae build from the final source tag and run the installed binary from Homebrew's install path.
- Package publication updates target-specific `npm-published`, `homebrew-formula-published`, or `homebrew-bottle-published` gates with package URL, version, checksum/provenance when available, source tag, actor, and command evidence.

Release note rules:

- Draft every binary manifest's `[release].notes_path`, then aggregate `releases/notes.md` when the release branch opens.
- Use discovered per-binary manifests as the source for binary names, versions, targets, package names, install commands, and asset names.
- Ask the release owner for the user-visible summary, notable changes, breaking changes, known issues, and audience-specific install guidance.
- Do not publish notes with placeholders, stale target/package names, or claims inferred only from commit subjects.
- All Markdown links in `releases/notes.md` and per-binary `releases/<bin>.md` must be absolute URLs. GitHub does not rewrite relative links in the rendered release body, so `[mmux](mmux.md)` 404s. Use `https://github.com/<owner>/<repo>/blob/<release-tag>/releases/<bin>.md` for per-binary notes; full details in `references/release-notes.md`.
- Before `gh release create`, confirm human approval and record final notes evidence in the workspace `github-release-published` gate.

Operator prompt workflow:

1. Read `WORKSPACE_MANIFEST`, then scan `releases/*.toml` for `kind = "motlie.binary-release"` manifests before proposing the next action.
2. Inspect the master issue, sub-issues, and sub-PRs when they exist; manifests remain authoritative if state conflicts.
3. Identify the first incomplete workspace gate or `(binary, gate, target_id)`, its required platform, and whether the current host can perform it.
4. If another host/operator is needed, create or update a scoped sub-issue that names the release branch, target manifest, required evidence, and expected sub-PR target.
5. If the action creates tags, GitHub Releases, npm publications, Homebrew tap changes, or closes the master issue, ask for explicit approval.
6. After a gate is performed, update manifest status on the release branch or through a sub-PR with `completed_at`, `completed_by`, `source_commit`, and `evidence`.
7. For handoffs, reply with current state, next gate, required host/platform, command group, files to update, issue/PR links, and approval needed.

Read references only when needed:

- npm publishing or token handling: `references/npm-auth.md`
- macOS `codesign` or Darwin artifact validation: `references/macos-signing.md`
- Homebrew formula/tap/bottle work: `references/homebrew-tap.md`
- release note drafting or validation: `references/release-notes.md`
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
- master issue:
- release branch:
- source tag:
- GitHub Release:
- sub-issues/sub-PRs:
- artifacts:
- macOS signing:
- npm:
- Homebrew:
- remaining gates:
```
