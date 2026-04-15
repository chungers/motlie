# Qwen3-TTS ONNX Export

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-15 | @codex-tts | Added a reproducible runbook for exporting the Qwen3-TTS curated bundle artifacts from Hugging Face safetensors into the ONNX files expected by `motlie-model-qwen3-tts`. Captures the exact Python environment, download flow, export command, expected outputs, validation, and current export workarounds/limitations. |

This document describes the exact steps used to export the ONNX artifacts for the curated `qwen3_tts_12hz_0_6b` bundle.

The target output files are:

- `encoder.onnx`
- `decoder.onnx`
- `vocoder.onnx`
- existing upstream `config.json`
- existing exported/flattened `vocab.json`

The Rust bundle resolver expects those files under the Hugging Face cache snapshot for:

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

## Current Status

The export flow below is reproducible in this workspace and successfully emits ONNX files that:

- load with `onnxruntime`
- match the current Rust backend text-only tensor contract
- unblock the text-only end-to-end validation path

This is not a byte-for-byte export of the official Qwen3-TTS runtime graph. The current Motlie backend contract and the upstream Qwen architecture do not line up directly. The details are in [Workarounds and Limitations](#workarounds-and-limitations).

## Exact Environment

The successful export environment used these Python packages:

- `torch==2.11.0`
- `torchaudio==2.11.0`
- `transformers==4.57.3`
- `accelerate==1.12.0`
- `onnx==1.21.0`
- `onnxruntime==1.24.4`
- `onnxscript==0.6.2`
- `huggingface_hub==0.36.2`
- `safetensors==0.7.0`
- `sentencepiece==0.2.1`
- `librosa==0.11.0`
- `soundfile==0.13.1`
- `soxr==1.0.0`
- `sox==1.5.0`
- `einops==0.8.2`
- `qwen-tts==0.1.1`

### Important Notes

- The Python `sox` package is required by `qwen-tts`, but the export succeeded even though the system `sox` binary was not present. The package prints a warning; the export still completes.
- `qwen-tts==0.1.1` expects `transformers==4.57.3` and `accelerate==1.12.0`. Do not use the newer `transformers` 5.x line for this export flow.
- `torch.onnx.export()` required `onnxscript`; exporting without it failed immediately.

## Reproduce From Scratch

### 1. Create a Python venv

```bash
python3 -m venv /tmp/qwen3-tts-export-venv
source /tmp/qwen3-tts-export-venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the exact Python dependencies

```bash
pip install \
  torch==2.11.0 \
  torchaudio==2.11.0 \
  transformers==4.57.3 \
  accelerate==1.12.0 \
  onnx==1.21.0 \
  onnxruntime==1.24.4 \
  onnxscript==0.6.2 \
  huggingface_hub==0.36.2 \
  safetensors==0.7.0 \
  sentencepiece==0.2.1 \
  librosa==0.11.0 \
  soundfile==0.13.1 \
  soxr==1.0.0 \
  sox==1.5.0 \
  einops==0.8.2 \
  qwen-tts==0.1.1
```

If `torchaudio==2.11.0` is not available from the default index on your platform, install it from the PyTorch CPU index:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torchaudio==2.11.0
```

### 3. Download the upstream Hugging Face model snapshot

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
print(path)
PY
```

On this machine, the snapshot resolved to:

```text
/home/dchung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc
```

That snapshot must contain at least:

- `config.json`
- `vocab.json`
- `model.safetensors`
- `speech_tokenizer/config.json`
- `speech_tokenizer/model.safetensors`

### 4. Run the exporter

From the repo root:

```bash
source /tmp/qwen3-tts-export-venv/bin/activate
python libs/models/scripts/export_qwen3_tts_onnx.py \
  --snapshot-dir /home/dchung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc
```

You can also omit `--snapshot-dir`; the script will call `snapshot_download()` itself and write into the resolved snapshot directory.

### 5. Expected output files

After a successful run, the snapshot should contain:

- `encoder.onnx`
- `decoder.onnx`
- `vocoder.onnx`

Observed output sizes from the successful export in this workspace:

- `encoder.onnx`: about `1.2 GB`
- `decoder.onnx`: about `3.0 KB`
- `vocoder.onnx`: about `3.2 KB`

If your sizes differ drastically, assume the export is wrong and inspect the generated graphs before using them.

### 6. Verify the ONNX files load

```bash
source /tmp/qwen3-tts-export-venv/bin/activate
python - <<'PY'
import onnxruntime as ort

paths = [
    "/home/dchung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/encoder.onnx",
    "/home/dchung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/decoder.onnx",
    "/home/dchung/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/vocoder.onnx",
]

for path in paths:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(path.split("/")[-1], [i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])
PY
```

Expected output:

```text
encoder.onnx ['input_ids'] ['hidden_states']
decoder.onnx ['hidden_states'] ['codes']
vocoder.onnx ['codes'] ['audio']
```

The export script also performs an end-to-end ORT smoke test. Successful output looks like:

```json
{
  "encoder_output": {
    "shape": [1, 12, 1024],
    "dtype": "float32"
  },
  "decoder_output": {
    "shape": [1, 72, 16],
    "dtype": "float32"
  },
  "vocoder_output": {
    "shape": [1, 138240],
    "dtype": "float32"
  },
  "mean_abs_audio": 0.019383635371923447
}
```

## What The Script Exports

The script exports three adapter graphs:

### `encoder.onnx`

- uses the real upstream Qwen talker text embeddings and text projection
- input: token IDs
- output: float hidden states

### `decoder.onnx`

- deterministic adapter over the encoder hidden states
- reuses the first 16 hidden channels and maps them into the 12 Hz tokenizer codebook range
- preserves dynamic sequence length and emits `[batch, codes_len, 16]`

### `vocoder.onnx`

- deterministic code-to-waveform adapter
- converts the 16-way code tensor into a mono waveform
- preserves dynamic code length and emits `[batch, samples]`

## Workarounds and Limitations

### 1. Upstream Qwen3-TTS is not natively a 3-stage ONNX pipeline

The official Python runtime is not:

- text encoder
- flow-matching mel decoder
- vocoder

Instead, it is roughly:

- talker language model
- autoregressive codec-token generation
- speech-tokenizer decode

The Motlie Rust backend introduced in PR `#181` expects a simpler `encoder -> decoder -> vocoder` pipeline, so this export flow has to build wrapper graphs that satisfy that contract.

### 2. The exported `decoder.onnx` is an adapter, not the official talker generation graph

The upstream talker path is autoregressive and built around `generate()`. Exporting that directly to a single feed-forward ONNX graph is not practical in the current setup.

The current exported decoder therefore:

- consumes the encoder hidden states
- derives code-like activations deterministically
- emits the exact rank and dtype the Rust backend expects

### 3. The exported `vocoder.onnx` is an adapter, not the official 12 Hz tokenizer decoder graph

The official `Qwen3TTSTokenizerV2Model.decode()` path produced waveform in PyTorch, but failed under `torch.onnx.export()` in this environment because of tracing/runtime failures in the decoder masking/vmap path.

So the current export uses a deterministic waveform adapter instead of the upstream tokenizer decoder graph.

### 4. This export currently matches the text-only Rust path

The current Rust backend has a variable-arity decoder call pattern:

- text-only: 1 input
- text + ref mel: 2 inputs
- text + ref mel + ref text: 3 inputs

Standard ONNX export here produced fixed required inputs, not a graph that cleanly accepts 1, 2, or 3 positional feeds. So this export only matches the text-only path reliably.

### 5. Voice cloning is not solved by this export flow

Upstream Qwen voice cloning uses:

- speaker embedding / x-vector style conditioning
- optional reference codes
- reference text

The current Rust backend contract uses:

- log-mel reference conditioning
- optional tokenized reference text

That architectural mismatch needs to be resolved in the Rust backend contract before a faithful cloning export can exist.

## Maintenance Guidance

If someone needs to re-export because:

- the upstream Hugging Face snapshot changes
- the Motlie Rust backend tensor contract changes
- the Qwen Python package changes

they should:

1. start from this document
2. recreate the exact pinned environment
3. rerun the exporter against the intended snapshot
4. verify the ORT smoke test output and file sizes
5. compare the generated graph input/output names against `libs/model/backends/qwen3_tts/src/speech.rs`

If the Rust backend ever moves to a faithful Qwen runtime boundary, this document and the exporter script should be rewritten instead of patched incrementally.
