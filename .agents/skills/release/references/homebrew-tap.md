# Homebrew Tap

Use this reference when updating Motlie Homebrew distribution.

Tap repo:

```text
github.com/motlie/homebrew-tap
```

User UX:

```sh
brew tap motlie/tap
brew install mmux
```

Formula path:

```text
Formula/mmux.rb
```

Preferred formula source:

```ruby
url "https://github.com/chungers/motlie/archive/refs/tags/v0.1.0.tar.gz"
sha256 "<source-tarball-sha256>"
```

Install shape:

```ruby
def install
  system "cargo", "build", "--release", "--locked", "-p", "motlie-mmux"
  bin.install "target/release/mmux"
  system "codesign", "--force", "--sign", "-", bin/"mmux" if OS.mac?
end
```

Test shape:

```ruby
test do
  assert_match "mmux", shell_output("#{bin}/mmux --version")
end
```

Homebrew release rule:

- update the tap by PR
- build and test bottles on macOS runners
- execute the installed bottled binary
- keep Homebrew bottle naming under Homebrew control; do not force npm/archive asset names onto bottles
