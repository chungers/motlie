# Homebrew Tap

Use this reference when updating Motlie Homebrew distribution.

Parameterize examples by the current per-binary manifest, not by workspace-level binary fields:

```text
BINARY_MANIFEST=releases/<bin>.toml
[identity].binary=<installed command name>
[identity].version=<release version>
[build].cargo_package=<cargo package name>
[build].cargo_bin=<cargo binary name>
[homebrew].formula=<formula name, often same as [identity].binary>
RELEASE_TAG=<YYYY-MM-codename>
```

Tap repo:

```text
github.com/motlie/homebrew-tap
```

User UX:

```sh
brew tap motlie/tap
brew install "<[homebrew].formula>"
```

Formula path:

```text
Formula/<formula>.rb
```

Preferred formula source:

```ruby
url "https://github.com/chungers/motlie/archive/refs/tags/<release-tag>.tar.gz"
sha256 "<source-tarball-sha256>"
```

Install shape:

```ruby
def install
  system "cargo", "build", "--release", "--locked", "-p", "<[build].cargo_package>", "--bin", "<[build].cargo_bin>"
  bin.install "target/release/<[identity].binary>"
  system "codesign", "--force", "--sign", "-", bin/"<[identity].binary>" if OS.mac?
end
```

Test shape:

```ruby
test do
  assert_match "<[identity].binary>", shell_output("#{bin}/<[identity].binary> --version")
end
```

Worked `mmux` example:

```text
BINARY_MANIFEST=releases/mmux.toml
[identity].binary=mmux
[identity].version=0.1.0
[build].cargo_package=motlie-mmux
[homebrew].formula=mmux
```

Homebrew release rule:

- update the tap by PR
- build from the final source tag after the release branch is finalized
- build and test bottles on macOS runners or manually for the first release
- execute the installed bottled binary
- keep Homebrew bottle naming under Homebrew control; do not force npm/archive asset names onto bottles
- update the retained Motlie release-branch ledger with the tap PR URL and tap commit after publication
