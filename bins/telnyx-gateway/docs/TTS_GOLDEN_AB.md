# TTS Golden A/B

## Changelog

- 2026-06-06 16:19 PDT, @codex-381-impl -- Added the reversed TTS quality A/B harness runbook for issue #381, covering Piper baseline, Kokoro-82M, Qwen3-TTS, codec round-trip scoring, objective audio signals, and M3.5 live-path interpretation.
- 2026-06-06 22:11 PDT, @codex-381-impl -- Updated Kokoro-82M from an external-command experiment to a curated in-repo ONNX backend that shares the workspace static ONNX Runtime link path.

## Purpose

This harness reverses the M1.5 golden ASR A/B workflow. The corpus text is fixed, each selected TTS engine synthesizes that text, the generated audio is normalized to 16 kHz mono PCM, round-tripped through the Telnyx `L16-16k` and `PCMU-8k` paths, and then transcribed by a fixed ASR backend. The default fixed ASR backend is `sherpa-2023` (`sherpa-zipformer-en-2023-06-26`). WER against the original text is used as the intelligibility proxy.

The runner also records objective signals for each synthesized sample:

- TTS synthesis elapsed time
- TTS realtime factor (`tts_elapsed_ms / audio_ms`)
- speaking rate in words per minute
- clipping percentage and peak absolute sample value
- ASR replay wall time after codec round-trip

MOS is not automated by this runner. Use the generated source WAVs and codec audit WAVs for a manual spot-check.

## Engines

Default engine order is:

1. `piper` (`piper/en_us_ljspeech_medium`) as the current practical baseline
2. `kokoro-82m` as the first candidate to beat Piper for M3.5 live full-duplex suitability
3. `qwen3-tts-cpp` as the second candidate, using the existing Motlie Qwen3-TTS C++ backend path

Kokoro-82M is a curated in-repo ONNX backend. It uses the workspace `ort`/`ort-sys` link path, so the full golden A/B binary must still link exactly one static ONNX Runtime and no dynamic `libonnxruntime`.

## Commands

Run a one-sample smoke against Piper with echo ASR to verify the runner shape without model downloads:

```bash
cargo run -p motlie-telnyx-gateway --features piper -- \
  --no-asr-download \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --engine piper \
  --asr-backend echo \
  --codec l16-16k \
  --limit 1 \
  --output-dir /tmp/motlie-tts-golden-ab-smoke \
  --output-json /tmp/motlie-tts-golden-ab-smoke.json
```

Run the full default matrix when Piper, Qwen3-TTS, Sherpa, and Kokoro artifacts are available:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --output-dir /tmp/motlie-tts-golden-ab-call-center \
  --output-json /tmp/motlie-tts-golden-ab-call-center.json
```

Run the PM/orchestration corpus with the same engines and codecs:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-pm-orchestration-golden.json \
  --output-dir /tmp/motlie-tts-golden-ab-pm \
  --output-json /tmp/motlie-tts-golden-ab-pm.json
```

Run only Kokoro first, which is the #381 priority candidate:

```bash
cargo run -p motlie-telnyx-gateway --features sherpa -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --engine kokoro-82m \
  --output-dir /tmp/motlie-tts-golden-ab-kokoro \
  --output-json /tmp/motlie-tts-golden-ab-kokoro.json
```

Kokoro artifacts resolve through the normal model artifact policy. The curated bundle expects `onnx/model_quantized.onnx`, `tokenizer.json`, and `voices/af_bella.bin` from `onnx-community/Kokoro-82M-v1.0-ONNX`.

## Output Layout

For each engine, source WAVs are written under:

```text
<output-dir>/<engine>/source/<sample-id>.wav
```

Codec round-trip ASR input WAVs are written under:

```text
<output-dir>/<engine>/asr-inputs/<codec>/<sample-id>.wav
```

The JSON report contains:

- `entries`: successful engine x codec x sample measurements
- `failures`: synthesis or ASR replay blockers, including missing model artifacts
- `summaries`: aggregate engine x codec x category rows plus an `ALL` category

## Recommendation Criteria

For M3.5 live full-duplex suitability, prefer an engine only if it improves or matches Piper intelligibility while staying viable on latency and throughput:

- lower aggregate WER over both `L16-16k` and `PCMU-8k`
- realtime factor comfortably below `1.0` on the target host, with margin for concurrent ASR and gateway work
- stable speaking rate for phone interaction, roughly in the normal conversational range
- near-zero clipping
- no repeated-token or tail artifacts in the generated and codec-round-tripped audit WAVs

If Kokoro beats Piper on WER and stays below realtime on the target host, it is the first recommendation candidate. If Kokoro is unavailable or too slow, Qwen3-TTS should be considered only if its quality gain outweighs the latency/throughput cost already observed in #188.
