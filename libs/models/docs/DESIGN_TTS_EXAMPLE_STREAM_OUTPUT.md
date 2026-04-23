# TTS Example Stream Output Design

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-21 | @codex-tts | Moved the generic WAV streaming and sample-conversion primitives out of `examples/` into a minimal shared `libs/voice` crate so the example layer now keeps only CLI/source-selection concerns. |
| 2026-04-21 | @codex-tts | Reworked stdout WAV emission so the examples no longer buffer the full utterance before writing. Stdout now uses an aligned indefinite-length RIFF/data header and writes chunks incrementally as they arrive from `SpeechStream`. |
| 2026-04-20 | @codex-tts | Removed `tts_qwen3_onnx` from the shipped example set in this PR after reconfirming it is non-functional for real speech output. The stream-output work now targets only the functional TTS examples: Piper and `qwen3-tts.cpp`. |
| 2026-04-21 | @codex-tts | Tightened `--quiet` so it suppresses backend-native stderr as well as example-layer diagnostics by redirecting process stderr during quiet example execution. |
| 2026-04-20 | @codex-tts | Added a shared `--quiet` flag for all shipped TTS examples so example-layer stderr diagnostics can be suppressed when stdout is carrying WAV bytes through a shell pipeline. |
| 2026-04-19 | @codex-tts | Revised the stdout design around simple shell composition. TTS examples now target streamed WAV on stdout instead of a custom framed PCM protocol so they can pipe directly into `ssh`, `afplay`, `ffmpeg`, or any ASR tool that accepts WAV from stdin. |
| 2026-04-19 | @codex-tts | Initial brownfield design for issue #208. Defines a uniform stdin/stdout pipeline mode for all shipped TTS example binaries while preserving `--wav` file output. |

This document defines the example-layer behavior for streamed output in the
shipped TTS binaries under `libs/models/examples/`.

Scope is intentionally narrow:

- `tts_piper`
- `tts_qwen3_tts_cpp`

The goal is to make those binaries behave the same at the CLI boundary without
changing the core `motlie_model` speech contract.

## Problem

The current TTS examples only support one sink:

- read text from `--text`
- synthesize whole speech output
- write a `.wav` file via `--wav`

That leaves two gaps:

1. the examples do not demonstrate streaming output through Unix pipes
2. the shipped binaries are already drifting in their argument surface

Because the examples are the public runnable UX for curated TTS bundles, they
need one shared contract rather than three backend-shaped CLIs.

## Goals

- Preserve current `--wav <path>` behavior.
- Add a pipeline mode that reads synthesis text from stdin.
- Emit a valid WAV byte stream on stdout so another process can consume it
  directly.
- Keep the shared flags and mode selection identical across all shipped TTS
  examples.
- Keep backend-specific flags additive and explicit, not implicit.

## Non-Goals

- Changing `motlie_model::typed::SpeechStream`.
- Adding telephony framing here.
- Adding playback to these binaries in this slice.
- Making every backend support every advanced flag. Only the common behavior
  must match.
- Defining a generic transport-neutral chunk protocol for stdout in this slice.

## Affected Binaries

The same feature applies to all current shipped TTS binaries:

1. `libs/models/examples/tts_piper/main.rs`
2. `libs/models/examples/tts_qwen3_tts_cpp/main.rs`

The implementation should also add shared example-layer code so future TTS
examples inherit the same behavior automatically instead of reimplementing CLI
and sink logic.

## Required Behavior

Every TTS example must support both modes:

### File Mode

If `--wav <path>` is present, the binary writes a `.wav` file to that path.
This remains the default and most user-friendly validation path.

The binary may accept text from either:

- `--text <value>`
- stdin when `--text` is omitted

This lets a caller pipe text in while still targeting `.wav`.

### Pipeline Mode

If `--wav` is not present, the binary:

- reads synthesis text from stdin unless `--text` is provided
- emits a valid WAV byte stream to stdout
- emits no human-readable progress lines on stdout

Human-readable logging must go to stderr in pipeline mode so stdout remains a
clean machine-readable WAV stream. `--quiet` suppresses both example-layer
diagnostics and backend-native stderr logging by redirecting process stderr to
`/dev/null` for the quiet execution window. This is intentionally
whole-process and best-effort: panic diagnostics may be silent while `--quiet`
is active.

This is the command-line composition target for this feature. The intended UX is
simple shell piping, for example:

```bash
echo "hello world" | cargo run -p motlie-models --example tts_piper \
  --no-default-features --features model-piper-en-us-ljspeech-medium \
  -- > out.wav
```

and remote/local playback, for example:

```bash
echo "hello world" | program | ssh mac-host '/opt/homebrew/bin/play -t wav -'
```

or:

```bash
echo "hello world" | program | ssh mac-host '/opt/homebrew/bin/play -t wav -'
```

## Proposed Shared CLI

Common flags for every TTS example:

- `--text <value>`: optional direct text input
- `--artifact-root <path>`: optional curated artifact root override
- `--wav <path>`: write a `.wav` file
- `--quiet`: suppress example-layer and backend-native stderr diagnostics

Common input rules:

- text source may be either `--text` or stdin
- if both stdin text and `--text` are provided, `--text` wins and stdin is
  ignored for this slice
- sink selection is:
  - `--wav <path>`: file output
  - no `--wav`: stdout WAV stream

Backend-specific flags remain allowed:

- `tts_qwen3_tts_cpp`: `--reference-audio`
- `tts_piper`: no backend-specific flags in the current slice

## Stdout WAV Contract

Stdout pipeline mode should emit a valid WAV container, not raw PCM and not a
custom chunk protocol.

Why:

- the user requirement is simple shell composition
- `ssh`, `afplay`, `ffmpeg`, and other CLI tools already understand WAV
- a TTS binary that writes WAV to stdout can also feed any ASR binary that
  accepts WAV from stdin
- it removes the need for a Motlie-specific adapter just to turn stdout into a
  file or player input

This means the example layer needs a stdout-safe WAV writer.

Important implementation constraints:

- `hound::WavWriter` requires `Write + Seek`
- stdout pipes are not seekable
- many stdin-side WAV readers, including `hound`, reject an indefinite data
  chunk length unless it is block-aligned
- so the examples need a small dedicated writer for stdout mode that emits an
  aligned indefinite-length WAV header and then writes audio bytes as they
  arrive

The file-path mode may keep using `hound` because a regular file is seekable.

The stdout-mode writer must:

- emit a correct WAV header for the backend output format
- use an aligned indefinite-length `RIFF`/`data` size so forward-only readers
  can consume until EOF without waiting for a full utterance buffer
- write samples incrementally as chunks arrive
- terminate cleanly at EOF without writing diagnostics to stdout

The initial implementation should target the current backend output shapes:

- Piper: mono `i16` at `22050 Hz`
- qwen3-tts.cpp: mono `f32` at backend-reported `24000 Hz`

## TTS to ASR Composition

Yes, this design intentionally supports piping a TTS example into an ASR binary
that accepts a WAV file stream from stdin.

That composition works if the ASR binary:

- reads WAV from stdin instead of requiring a filesystem path
- is willing to consume a forward-only stream until EOF

The matching ASR example work in issue #208 adds a tolerant stdin-side WAV
parser so the shipped ASR examples can consume the aligned indefinite-length
header emitted here.

## Layering

This feature belongs in the example layer, not the backend layer.

The core speech model contract already yields PCM chunks. The examples should:

- parse CLI
- choose text source
- choose sink
- translate `SpeechStream` output into either:
  - `.wav` file bytes through the existing file writer path
  - stdout WAV bytes through a dedicated non-seekable writer

That implies adding a small shared helper module under `libs/models/examples/`
for:

- common TTS argument parsing
- stdin text loading
- example-layer file/stdout sink selection

The reusable audio mechanics themselves now live in `libs/voice`:

- `motlie_voice::wav::StreamingWavWriter`
- `motlie_voice::wav::WavSample`

That keeps the example layer focused on CLI behavior instead of duplicating
generic audio code.

## Error Handling

The binaries must:

- return a non-zero exit code on invalid argument combinations
- fail clearly when stdin is required but empty
- fail clearly when stdout stream writing errors
- keep stdout free of diagnostic logging in pipeline mode

## Validation

The implementation should prove:

1. each binary still writes `.wav`
2. each binary can read text from stdin
3. each binary can emit a valid WAV stream on stdout
4. the stdout WAV stream can be consumed by standard CLI tools
5. backend-specific flags still work where supported

## Alternatives Considered

### Raw PCM on stdout

Rejected. Downstream tools would not know sample rate, encoding, or end of
stream without out-of-band coordination.

### Custom framed PCM protocol on stdout

Rejected for this issue. It solves a more general transport problem than the
current requirement and makes simple shell composition worse. Telnyx or other
transport adapters can be added later on top of the same speech stream contract
without making the TTS examples speak a custom stdout protocol now.

### New core-model stream type

Rejected for this issue. The model layer already exposes enough information.
The inconsistency is in the examples, not the speech contract.

### Implement only one binary first

Rejected. The issue explicitly applies to all shipped TTS examples, and partial
alignment would keep the public CLI surface inconsistent.
