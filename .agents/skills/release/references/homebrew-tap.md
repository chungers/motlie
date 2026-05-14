# Homebrew Tap

Use this reference when updating Motlie Homebrew distribution.

Parameterize examples by the release target:

```text
BIN=<installed command name>
CARGO_PACKAGE=<cargo package name>
FORMULA=<formula name, often same as BIN>
VERSION=<release version>
BINARY_MANIFEST=releases/<bin>-<version>.toml
RELEASE_TAG=<YYYY-MM-codename>
```

Tap repo:

```text
github.com/motlie/homebrew-tap
```

User UX:

```sh
brew tap motlie/tap
brew install "${FORMULA}"
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
  system "cargo", "build", "--release", "--locked", "-p", "<cargo-package>"
  bin.install "target/release/<bin>"
  system "codesign", "--force", "--sign", "-", bin/"<bin>" if OS.mac?
end
```

Test shape:

```ruby
test do
  assert_match "<bin>", shell_output("#{bin}/<bin> --version")
end
```

Worked `mmux` example:

```text
BIN=mmux
CARGO_PACKAGE=motlie-mmux
FORMULA=mmux
```

Homebrew release rule:

- update the tap by PR
- build from the final source tag after the release branch is finalized
- build and test bottles on macOS runners or manually for the first release
- execute the installed bottled binary
- keep Homebrew bottle naming under Homebrew control; do not force npm/archive asset names onto bottles
- update the retained Motlie release-branch ledger with the tap PR URL and tap commit after publication
