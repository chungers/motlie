# Release Checklist

Use this checklist when coordinating an end-to-end Motlie release.

- [ ] Release target captured: `BIN`, `CARGO_PACKAGE`, `CARGO_BIN`, `VERSION`, channels, and targets.
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
- [ ] npm auth path selected: trusted publishing or temporary `NPM_TOKEN`.
- [ ] npm packages published to `@motlie`.
- [ ] npm installs verified for Linux and macOS packages.
- [ ] Homebrew tap PR opened against `motlie/homebrew-tap`.
- [ ] Homebrew formula builds from source tag.
- [ ] Homebrew formula re-signs installed binary on macOS.
- [ ] Homebrew bottle tests pass from installed path.
- [ ] Homebrew tap PR merged.
- [ ] GitHub Release marked stable.
