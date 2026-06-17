# TTS Golden A/B

## Changelog

- 2026-06-15 PDT, @codex-m6-ds-rv -- Documented that Kokoro streaming warmup expects the curated `motlie-models-download kokoro_82m` flow to generate Sherpa-compatible `tokens.txt` from `tokenizer.json`; the gateway still never downloads or repairs artifacts at runtime.
- 2026-06-06 16:19 PDT, @codex-381-impl -- Added the reversed TTS quality A/B harness runbook for issue #381, covering Piper baseline, Kokoro-82M, Qwen3-TTS, codec round-trip scoring, objective audio signals, and M3.5 live-path interpretation.
- 2026-06-06 22:11 PDT, @codex-381-impl -- Updated Kokoro-82M from an external-command experiment to a curated in-repo ONNX backend that shares the workspace static ONNX Runtime link path.
- 2026-06-06 22:50 PDT, @codex-381-impl -- Made Qwen3-TTS an explicit opt-in golden-ab lane after DGX review found a single runaway sample could generate hundreds of seconds of audio and retain high RSS; default bakeoff is now Piper + Kokoro only.

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

`qwen3-tts-cpp` remains available as an explicit `--engine qwen3-tts-cpp` lane, but it is no longer part of the implicit default matrix. DGX review found a Qwen3 runaway sample that generated hundreds of seconds of audio, took several minutes, and retained high RSS. Keep Qwen3 opt-in until that backend has a hard per-sample output-duration/timeout guard.

Kokoro-82M is a curated in-repo ONNX backend. It uses the workspace `ort`/`ort-sys` link path, so the full golden A/B binary must still link exactly one static ONNX Runtime and no dynamic `libonnxruntime`.

## Commands

Run a one-sample smoke against Piper with echo ASR after preloading model artifacts:

```bash
cargo run -p motlie-telnyx-gateway --features piper -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --engine piper \
  --asr-backend echo \
  --codec l16-16k \
  --limit 1 \
  --output-dir /tmp/motlie-tts-golden-ab-smoke \
  --output-json /tmp/motlie-tts-golden-ab-smoke.json
```

Run the full default matrix after Piper, Sherpa, and Kokoro artifacts are preloaded:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --output-dir /tmp/motlie-tts-golden-ab-call-center \
  --output-json /tmp/motlie-tts-golden-ab-call-center.json
```

Run the PM/orchestration corpus with the same default engines and codecs:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-pm-orchestration-golden.json \
  --output-dir /tmp/motlie-tts-golden-ab-pm \
  --output-json /tmp/motlie-tts-golden-ab-pm.json
```

Run Qwen3-TTS only when explicitly investigating that lane and when the host can tolerate a possible long sample; this is not part of the default matrix:

```bash
cargo run -p motlie-telnyx-gateway --features golden-ab -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --engine qwen3-tts-cpp \
  --output-dir /tmp/motlie-tts-golden-ab-qwen3 \
  --output-json /tmp/motlie-tts-golden-ab-qwen3.json
```

Run only Kokoro first, which is the #381 priority candidate:

```bash
cargo run -p motlie-telnyx-gateway --features sherpa -- \
  tts-golden-ab bins/telnyx-gateway/corpus/qwen3-call-center-golden.json \
  --engine kokoro-82m \
  --output-dir /tmp/motlie-tts-golden-ab-kokoro \
  --output-json /tmp/motlie-tts-golden-ab-kokoro.json
```

Kokoro artifacts must be preloaded before running this gateway command; the gateway does not download model artifacts. The curated bundle expects `onnx/model_quantized.onnx`, `tokenizer.json`, and `voices/af_bella.bin` from `onnx-community/Kokoro-82M-v1.0-ONNX`. For streaming mode, run the curated `motlie-models-download kokoro_82m` flow so `tokens.txt` is generated from Kokoro `tokenizer.json` in the snapshot before warmup.

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
