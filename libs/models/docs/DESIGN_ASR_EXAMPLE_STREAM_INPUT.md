# ASR Example Stream Input Design

## Status: Proposed

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-21 | @codex-tts | Moved the generic WAV decode/downmix/resample primitives out of `examples/` into a minimal shared `libs/voice` crate, and collapsed the two streaming ASR mains onto a shared streaming runner helper. |
| 2026-04-21 | @codex-tts | Replaced the earlier “plain `hound` on stdin” assumption with a tolerant example-layer stdin WAV parser so the ASR examples can consume the aligned indefinite-length WAV headers emitted by the TTS stdout pipeline. |
| 2026-04-21 | @codex-tts | Tightened `--quiet` so it suppresses backend-native stderr as well as example-layer diagnostics by redirecting process stderr during quiet example execution. |
| 2026-04-20 | @codex-tts | Refined the stdout contract after live TTS→ASR pipe validation. ASR examples now default to one final plain-text transcript line on stdout, add `--partials` for streaming event output, and add `--quiet` to suppress example-layer stderr diagnostics. |
| 2026-04-19 | @codex-tts | Initial brownfield design for stdin WAV support in all shipped ASR examples so they compose directly with the TTS example stdout WAV contract from issue #208. |

This document defines the example-layer behavior for streamed WAV input in the
shipped ASR binaries under `libs/models/examples/`.

Scope is intentionally narrow:

- `asr_whisper`
- `asr_sherpa_onnx`
- `asr_moonshine`

The goal is to make those binaries consume either a `.wav` file path or a WAV
byte stream from stdin while keeping transcript output on stdout.

## Problem

The current ASR examples only support one input source:

- `--wav <path>` points at a filesystem path

That blocks direct shell composition with the TTS examples. Even after the TTS
examples can emit WAV on stdout, the ASR side still cannot accept that stream
without an intermediate temp file.

## Goals

- Preserve current `--wav <path>` behavior.
- Allow WAV input from stdin when `--wav` is omitted.
- Keep transcript text on stdout.
- Keep diagnostics on stderr when stdin/stdout are used as a machine pipeline.
- Keep the shared input/output behavior identical across all shipped ASR
  examples.

## Non-Goals

- Changing the core typed ASR contracts in `motlie_model`.
- Adding microphone capture in this slice.
- Defining a structured JSON transcript protocol in this slice.
- Making batch Whisper behave like true incremental streaming. It remains a
  batch transcription backend even when its WAV arrives over stdin.

## Affected Binaries

The same feature applies to all current shipped ASR binaries:

1. `libs/models/examples/asr_whisper/main.rs`
2. `libs/models/examples/asr_sherpa_onnx/main.rs`
3. `libs/models/examples/asr_moonshine/main.rs`

The implementation should add shared example-layer helpers so future ASR
examples inherit the same input/output behavior automatically.

## Required Behavior

Every ASR example must support both input modes:

### File Mode

If `--wav <path>` is present, the binary reads the WAV file from that path.
This remains the default explicit path-based mode.

### Pipeline Mode

If `--wav` is omitted, the binary reads a WAV byte stream from stdin.

This is the command-line composition target for this feature. The intended UX
is:

```bash
echo "hello world" | tts_program | asr_program
```

and also:

```bash
cat input.wav | asr_program
```

In pipeline mode:

- stdin is treated as binary WAV input
- stdout is reserved for transcript text
- diagnostics must go to stderr

## Proposed Shared CLI

Common flags for every ASR example:

- `--wav <path>`: optional file input override
- `--artifact-root <path>`: optional curated artifact root override
- `--quiet`: suppress example-layer and backend-native stderr diagnostics

Backend-specific flags remain additive:

- `asr_whisper`: `--language <code>`
- `asr_sherpa_onnx`: `--partials`
- `asr_moonshine`: `--partials`

Common input rules:

- `--wav <path>` means read that file
- no `--wav` means read WAV from stdin
- empty stdin should fail clearly

## Stdout Transcript Contract

Stdout should stay simple in this slice: plain text transcript lines, not a new
container format.

The examples may continue to print their current text lines such as `[final]`
or `[partial]`, but machine-readable transcript output must not be mixed with
diagnostics on stdout in pipeline mode.

The default stdout behavior is one final transcript line with no markers or
timestamps. That keeps shell composition simple:

```bash
echo "hello world" | tts_program | asr_program
```

Streaming event output remains opt-in through `--partials` for the streaming
backends. In that mode the examples may print `[partial]` / `[final]` event
lines to stdout, but that is no longer the default CLI behavior.

That means:

- final transcript text stays on stdout
- progress, format, and artifact-path logging move to stderr when using stdin
  or stdout in a pipeline
- `--quiet` suppresses both example-layer diagnostics and backend-native stderr
  logging by redirecting process stderr to `/dev/null` for the quiet execution
  window
- as on the TTS side, this is intentionally whole-process and best-effort:
  panic diagnostics may be silent while `--quiet` is active

This keeps shell composition easy while avoiding a new protocol decision.

## Feasibility

This is practical with the current crate stack.

Important implementation details:

- `hound::WavWriter` requires `Seek`, which is why TTS stdout needed special
  handling
- the TTS stdout path now emits an aligned indefinite-length `data` chunk so it
  can start writing immediately without buffering the full utterance
- `hound::WavReader` does not tolerate that header shape cleanly on stdin, so
  the ASR examples use a small example-layer parser for stdin mode instead of
  passing stdin directly to `hound`

File-path mode still uses `hound` normally. Stdin mode uses the tolerant parser
only so the example binaries can compose directly with the TTS stdout contract.

## Backend-Specific Notes

### Whisper

`asr_whisper` is still a batch backend. In stdin mode it will read and decode
the entire WAV stream, normalize it to `AudioBuf<f32, 16000, Mono>`, and only
then call `transcribe(...)`.

That is still useful for shell composition, but it is not low-latency
incremental transcription.

### Sherpa ONNX

`asr_sherpa_onnx` already operates as a streaming session after audio decode.
In this slice, stdin support means:

- decode WAV from stdin through the tolerant example-layer parser
- normalize to 16 kHz mono
- ingest fixed-size demo chunks into the typed session

### Moonshine

`asr_moonshine` follows the same source adaptation pattern as sherpa:

- decode WAV from stdin through the tolerant example-layer parser
- normalize to 16 kHz mono
- ingest fixed-size chunks into the typed session

## Layering

This belongs in the example layer, not the backend layer.

The backends already consume typed audio buffers. The examples should:

- parse CLI
- choose WAV source: file or stdin
- decode and normalize WAV
- print transcript text to stdout
- keep diagnostics on stderr for pipeline safety

That implies adding shared example-layer support for:

- common ASR argument parsing
- stdin/file WAV source selection
- stderr-safe logging decisions

The generic audio work now lives below the example layer in `libs/voice`:

- `motlie_voice::pipeline::convert`
- `motlie_voice::pipeline::resample`
- `motlie_voice::wav`

The sherpa and Moonshine mains then share one example-layer streaming runner on
top of those primitives instead of maintaining near-identical ingest loops.

## Error Handling

The binaries must:

- return a non-zero exit code on invalid arguments
- fail clearly on empty or invalid stdin WAV input
- fail clearly when WAV decode fails
- keep diagnostics off stdout in pipeline mode

## Validation

The implementation should prove:

1. each binary still accepts `--wav <path>`
2. each binary accepts WAV from stdin when `--wav` is omitted
3. transcript text remains on stdout
4. diagnostics move to stderr in pipeline mode
5. `tts_example | asr_example` works for at least one paired pipeline

## Alternatives Considered

### Keep ASR examples path-only

Rejected. That would keep the shell composition story incomplete and force temp
files even after TTS stdout improves.

### Add a new transcript framing protocol

Rejected for this slice. Plain text on stdout is enough for command-line use and
does not block a later richer protocol if needed.
