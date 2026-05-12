# Homebrew Tap

Use this reference when updating Motlie Homebrew distribution.

Parameterize examples by the release target:

```text
BIN=<installed command name>
CARGO_PACKAGE=<cargo package name>
FORMULA=<formula name, often same as BIN>
VERSION=<release version>
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
url "https://github.com/chungers/motlie/archive/refs/tags/v#{version}.tar.gz"
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
- build and test bottles on macOS runners
- execute the installed bottled binary
- keep Homebrew bottle naming under Homebrew control; do not force npm/archive asset names onto bottles
