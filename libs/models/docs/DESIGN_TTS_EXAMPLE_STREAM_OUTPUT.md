# TTS Example Stream Output Design

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-19 | @codex-tts | Initial brownfield design for issue #208. Defines a uniform stdin/stdout pipeline mode for all shipped TTS example binaries while preserving `--wav` file output. |

This document defines the example-layer behavior for streamed output in the
shipped TTS binaries under `libs/models/examples/`.

Scope is intentionally narrow:

- `tts_piper`
- `tts_qwen3_onnx`
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
2. the three binaries are already drifting in their argument surface

Because the examples are the public runnable UX for curated TTS bundles, they
need one shared contract rather than three backend-shaped CLIs.

## Goals

- Preserve current `--wav <path>` behavior.
- Add a pipeline mode that reads synthesis text from stdin.
- Emit framed stream output on stdout so another process can consume it.
- Keep the shared flags and mode selection identical across all shipped TTS
  examples.
- Keep backend-specific flags additive and explicit, not implicit.

## Non-Goals

- Changing `motlie_model::typed::SpeechStream`.
- Adding telephony framing here.
- Adding playback to these binaries in this slice.
- Making every backend support every advanced flag. Only the common behavior
  must match.

## Affected Binaries

The same feature applies to all current shipped TTS binaries:

1. `libs/models/examples/tts_piper/main.rs`
2. `libs/models/examples/tts_qwen3_onnx/main.rs`
3. `libs/models/examples/tts_qwen3_tts_cpp/main.rs`

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

If `--stdout-stream` is present, the binary:

- reads synthesis text from stdin unless `--text` is provided
- emits a framed stream to stdout
- emits no human-readable progress lines on stdout

Human-readable logging must go to stderr in pipeline mode so stdout remains a
clean machine-readable stream.

If neither `--wav` nor `--stdout-stream` is given, argument parsing should fail
with a clear error because the sink is ambiguous.

If both are given, the initial slice should reject that combination to keep the
first implementation simple and deterministic.

## Proposed Shared CLI

Common flags for every TTS example:

- `--text <value>`: optional direct text input
- `--artifact-root <path>`: optional curated artifact root override
- `--wav <path>`: write a `.wav` file
- `--stdout-stream`: write framed PCM chunks to stdout

Common input rules:

- exactly one sink must be selected: `--wav` or `--stdout-stream`
- text source may be either `--text` or stdin
- if both stdin text and `--text` are provided, `--text` wins and stdin is
  ignored for this slice

Backend-specific flags remain allowed:

- `tts_qwen3_onnx`: `--reference-audio`, `--reference-text`
- `tts_qwen3_tts_cpp`: `--reference-audio`
- `tts_piper`: no backend-specific flags in the current slice

## Stream Framing Contract

Stdout pipeline mode needs an explicit framing layer. Raw PCM by itself is not
enough because downstream tools need the audio format and chunk boundaries.

The proposed initial framing is:

1. one UTF-8 JSON header line
2. repeated binary chunk frames

Header example:

```json
{"version":1,"encoding":"pcm_s16le","sample_rate_hz":22050,"channels":1}
```

Header rules:

- newline-delimited UTF-8 JSON
- emitted exactly once before any audio frame
- describes the PCM payload format for every subsequent frame

Chunk frame rules:

- 4-byte little-endian unsigned payload length
- exactly that many PCM bytes immediately after the length field
- zero-length payload marks end of stream

Why this design:

- simple to generate from Rust examples
- simple to consume from another Rust tool or shell adapter
- stable enough to support a later `stdout-stream -> wav` adapter
- avoids mixing binary PCM with human-readable progress text

## Layering

This feature belongs in the example layer, not the backend layer.

The core speech model contract already yields PCM chunks. The examples should:

- parse CLI
- choose text source
- choose sink
- translate `SpeechStream` output into either `.wav` or framed stdout bytes

That implies adding a small shared helper module under `libs/models/examples/`
for:

- common TTS argument parsing
- stdin text loading
- `.wav` sink writing
- stdout framing

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
3. each binary can emit framed stdout stream output
4. a simple adapter can consume that stdout stream and write a `.wav`
5. backend-specific flags still work where supported

## Alternatives Considered

### Raw PCM on stdout without framing

Rejected. Downstream tools would not know sample rate, encoding, or end of
stream without out-of-band coordination.

### New core-model stream type

Rejected for this issue. The model layer already exposes enough information.
The inconsistency is in the examples, not the speech contract.

### Implement only one binary first

Rejected. The issue explicitly applies to all shipped TTS examples, and partial
alignment would keep the public CLI surface inconsistent.
