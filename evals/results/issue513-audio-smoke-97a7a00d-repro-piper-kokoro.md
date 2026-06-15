# Issue 513 Piper/Kokoro CUDA Repro Rerun

- implementer: `@513-impl`
- build_sha: `97a7a00d2ade8b31c284b53036268184e2e96591`
- host: `spark-2f6e`
- arch: `aarch64`
- CUDA target: `sm_121`, CUDA 13, cuDNN 9
- ORT link policy: static; ONNX Runtime source: `libs/model/backends/ort/vendor/onnxruntime@v1.24.2`
- review fix: removed hidden `MOTLIE_PIPER_ALLOW_CUDA` / `MOTLIE_KOKORO_ALLOW_CUDA` opt-in gates; piper/kokoro now follow the compiled CUDA feature and requested accelerator path like sherpa/moonshine.
- rerun env: both `MOTLIE_PIPER_ALLOW_CUDA` and `MOTLIE_KOKORO_ALLOW_CUDA` were unset for every row.
- result dirs:
  - `evals/results/cold/issue513-repro-audio-smoke-97a7a00d-spark-2f6e-aarch64-cuda`
  - `evals/results/warm/issue513-repro-audio-smoke-97a7a00d-spark-2f6e-aarch64-cuda`

All rerun rows passed with `requested_accelerator=cuda`, `resolved_accelerator=cuda`, and backend modes `piper:cuda` / `kokoro:cuda`. The recorded `runtime.cargo_features` include `model-kokoro-82m`, `kokoro-cuda`, and `moonshine-cuda`.

## Cold

| engine | resolved | backend | ttfa_ms | request_ms | startup_ms | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| piper | `cuda` | `piper:cuda` | 56.0 | 56.0 | 1325 | `pass` |
| kokoro | `cuda` | `kokoro:cuda` | 947.0 | 947.0 | 806 | `pass` |

## Warm

| engine | resolved | backend | mean_ttfa_ms | p95_ttfa_ms | mean_request_ms | startup_ms | warmup_ms | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| piper | `cuda` | `piper:cuda` | 47.7 | 54.0 | 47.7 | 1142 | 58 | `pass` |
| kokoro | `cuda` | `kokoro:cuda` | 909.0 | 916.0 | 909.0 | 819 | 959 | `pass` |
