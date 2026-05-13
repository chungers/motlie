# Release Checklist

Use this checklist when coordinating an end-to-end Motlie release.

- [ ] Release target captured: `BIN`, `CARGO_PACKAGE`, `CARGO_BIN`, `VERSION`, channels, targets, `MANIFEST`, and `RELEASE_BRANCH`.
- [ ] Release skill inspected `MANIFEST` and identified the next incomplete gate before taking action.
- [ ] Human prompt included current state, next gate, required platform, branch or PR to pull, files to update, and approval needed.
- [ ] Release coordination branch created from `main`.
- [ ] `releases/<bin>/<version>.toml` committed with deterministic intent, explicit names, target matrix, and gates.
- [ ] `releases/<bin>/<version>.md` committed as the release-note source.
- [ ] Coordination PR opened against `main`.
- [ ] Platform/channel sub-PRs opened against the release branch as needed.
- [ ] Sub-PRs update manifest status with source commit, timestamp, actor, and evidence links.
- [ ] Each gate handoff is reconstructable from manifest evidence and PR comments.
- [ ] Build outputs are not committed to git.
- [ ] Manifest confirms native npm mode when required, for example `runner = "native-binary"` and `node_launcher = false`.
- [ ] macOS signing evidence is recorded for Darwin targets that require it.
- [ ] Coordination PR is up to date with `main`.
- [ ] Coordination PR merged to `main`.
- [ ] Final source tag pushed from `main`.
- [ ] Final artifacts built from the final source tag.
- [ ] Final Darwin artifacts signed and verified from the installed path.
- [ ] Final checksums generated.
- [ ] GitHub Release published with final archives, checksums, installer assets, and notes.
- [ ] Direct installer verified from release-pinned URL.
- [ ] npm packages generated from final artifacts.
- [ ] `npm pack --dry-run` reviewed for each package.
- [ ] npm auth path selected: trusted publishing or temporary `NPM_TOKEN`.
- [ ] npm packages published to `@motlie`.
- [ ] npm installs verified for Linux and macOS packages.
- [ ] Homebrew tap PR opened against `motlie/homebrew-tap`.
- [ ] Homebrew formula builds from final source tag.
- [ ] Homebrew formula re-signs installed binary on macOS.
- [ ] Homebrew bottle tests pass from installed path.
- [ ] Homebrew tap PR merged.
- [ ] Post-release ledger PR updates `releases/<bin>/<version>.toml` to `state = "published"` with final URLs, checksums, npm links, Homebrew tap commit, and install evidence.
