# Telnyx Gateway Release Build Notes

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-07 | @codex-366-impl | Added release-build guidance that treats Cargo features as part of the gateway release contract, including Kokoro default TTS, Piper fallback TTS, and Sherpa static ONNX Runtime verification. |

This document records release-specific build checks for `telnyx-gateway`.
Operator docs and code defaults are not enough: the release binary must be built
with the Cargo features that make those defaults available at runtime.

## Release Feature Contract

The gateway package has `default = []`, so every release build must pass an
explicit feature set.

For the current live full-duplex gateway, the minimal release feature set is:

```text
sherpa + kokoro + piper
```

Feature responsibilities:

| Feature | Runtime role | Release requirement |
|---------|--------------|---------------------|
| `sherpa` | Live ASR backend family for Telnyx calls. | Required for the current live ASR path. |
| `kokoro` | Default live TTS backend, exposed as `kokoro-82m`. | Required whenever `LiveTtsBackend::default()` is `kokoro-82m`. |
| `piper` | Fallback live TTS backend. | Required while the operator UX and docs advertise Piper fallback. |

Do not ship a release binary built with only `sherpa piper` while the code
default remains `kokoro-82m`. That binary starts with an unavailable default TTS
backend and reports:

```text
status=unavailable
reason=Kokoro-82M TTS is unavailable; rebuild with --features kokoro
```

If the code default changes, update this release feature contract in the same
PR. If the fallback changes, update the contract and operator docs together.

## Build Commands

Release build:

```sh
cargo build --release -p motlie-telnyx-gateway --features "sherpa kokoro piper"
```

Validation gates for gateway release candidates:

```sh
cargo test -p motlie-telnyx-gateway --features "sherpa kokoro piper"
cargo clippy -p motlie-telnyx-gateway --features "sherpa kokoro piper" -- -D warnings
```

Package-scoped formatting check:

```sh
cargo fmt -p motlie-telnyx-gateway --check
```

## Runtime Smoke Check

Start the TUI smoke-test gateway with the local Telnyx test config:

```sh
cd /home/dchung/sessions/issue-358-telnyx-voice/codex-366-m3

set -a
source /home/dchung/telnyx-test/telnyx.env
set +a

./target/release/telnyx-gateway \
  --tui \
  --conversation-smoke-test \
  --load /home/dchung/telnyx-test/config.repl \
  --socket /tmp/telnyx-m3-live.sock
```

In the TUI, run:

```text
tts status
```

Release-safe expected result:

```text
next=kokoro-82m
default=kokoro-82m
available=kokoro-82m,piper
status=available
```

The exact line ordering may vary with the command renderer, but
`kokoro-82m` must be available and selected as the default. Piper must remain
listed as an available fallback.

## Static-Linkage Nuance

The Linux release binary is not currently a fully static executable. It is
expected to depend on system libraries such as OpenSSL, libstdc++, libm,
libgcc_s, libc, and the dynamic loader.

Sherpa's ONNX Runtime dependency is different: the upstream `sherpa-onnx` Rust
crate downloads and statically links its native archive, including the
`onnxruntime.a` static library used internally by Sherpa. A release binary
should not have a dynamic `libonnxruntime.so` dependency.

Check the dynamic dependencies with:

```sh
file target/release/telnyx-gateway
ldd target/release/telnyx-gateway
readelf -d target/release/telnyx-gateway
```

Expected:

- `file` may report a dynamically linked ELF.
- `ldd` may list system libraries.
- `readelf -d` must not list `libonnxruntime.so` in `NEEDED`.

Do not set `ORT_LIB_PATH`, `ORT_LIB_LOCATION`, `ORT_PREFER_DYNAMIC_LINK`, or
`LD_LIBRARY_PATH` for the Sherpa live-test path.

## Release Checklist

- [ ] Build with `--features "sherpa kokoro piper"`.
- [ ] Run gateway package tests with the same feature set.
- [ ] Run gateway clippy with the same feature set and `-D warnings`.
- [ ] Run package-scoped `cargo fmt --check`.
- [ ] Verify `tts status` reports `kokoro-82m` available and default.
- [ ] Verify Piper is listed as available fallback.
- [ ] Verify no dynamic `libonnxruntime.so` dependency appears in `readelf -d`.

