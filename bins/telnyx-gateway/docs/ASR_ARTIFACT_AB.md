# Telnyx ASR Artifact A/B Report

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-02 | @codex-371-impl | Added the piece-3 Sherpa artifact A/B setup and recorded the current golden-WAV blocker on this host. |

## Scope

Piece 3 compares Sherpa artifacts behind the existing Telnyx ASR adapter and `replay-corpus` harness. It does not add hotwords, contextual bias, endpointing changes, decoder tuning, post-ASR normalization, or a new ASR backend.

## Candidates

| Replay backend value | Artifact | Source |
|---|---|---|
| `sherpa-zipformer-2023` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26` | Current Motlie Sherpa streaming Zipformer artifact. |
| `sherpa-zipformer-kroko-2025` | `csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Newer Sherpa-supported English streaming Zipformer artifact. Hugging Face lists `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, and `tokens.txt` for this repo: https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/tree/main |

## Golden Corpus Status

Blocked on this host as of 2026-06-02 PDT.

Observed:

- `bins/telnyx-gateway/corpus/asr-golden.json` is present.
- `bins/telnyx-gateway/corpus/external/...` contains no capture directories or WAVs.
- `find /home/dchung -name asr-input-16khz.wav -print` returned no files.
- `find /home/dchung -name manifest.json -path '*telnyx*captures*' -print` returned no Telnyx capture manifests.
- The exact outbound 65-word reference file for the `29.2%` L16 baseline is not present on this host.

Required before scoring:

- Copy the 29.2% L16 `16 kHz` capture directory and exact 65-word reference into the manifest paths, or update a local manifest copy to point at those files.
- Copy at least one PCMU `8 kHz` capture directory with `asr-input-16khz.wav` into the manifest PCMU slot, or update a local manifest copy.

## Command

```sh
cargo run -p motlie-telnyx-gateway --features sherpa --    replay-corpus bins/telnyx-gateway/corpus/asr-golden.json   --backend sherpa-zipformer-2023   --backend sherpa-zipformer-kroko-2025   --chunk-ms 20
```

Expected report fields per corpus entry and artifact: raw transcript, WER, substitutions, deletions, insertions, token errors, chunk/audio metadata, and latency (`audio_ms`, ingest total/avg/max, finish, wall).

## Results

No WER or latency comparison is available yet because the golden WAV/reference artifacts are absent on this host. Model artifacts must also be preloaded; the gateway command does not download them. The replay command fails until the external corpus files are populated.

Observed command result on 2026-06-02 PDT:

```text
Error: read reference transcript bins/telnyx-gateway/corpus/references/telnyx-outbound-l16-2026-06-01-wer29.txt

Caused by:
    No such file or directory (os error 2)
```

## #191-Gated Follow-Ups

Moonshine, Nemotron, and Parakeet A/B remain gated on #191 backend/model work. They should be added to this report only after those backends land and can run through the same raw-ASR replay corpus.
