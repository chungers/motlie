# Issue 513 DGX Audio CUDA vs CPU

- implementer: `@513-impl`
- build_sha: `cb2ec5f15db85bc57f9e124da7ff06d6e8db9578`
- host: `spark-2f6e`
- arch: `aarch64`
- CUDA target: `sm_121`, CUDA 13, cuDNN 9
- CUDA ORT source: `libs/model/backends/ort/vendor/onnxruntime@v1.24.2`
- ORT link policy: static; runtime dynamic CUDA dependency observed by `ldd`: `libcudnn.so.9`

All rows passed behavior, resource, performance, and accelerator gates. CUDA rows recorded `requested_accelerator=cuda` and `resolved_accelerator=cuda`; CPU rows recorded `requested_accelerator=cpu` and `resolved_accelerator=cpu`.

## Cold

| engine | metric | CUDA | CPU | CUDA backend | CPU backend |
| --- | ---: | ---: | ---: | --- | --- |
| sherpa | ttfp_ms | 147.0 | 130.0 | `sherpa_onnx:cuda` | `sherpa_onnx:cpu` |
| sherpa | request_ms | 648.0 | 581.0 | `sherpa_onnx:cuda` | `sherpa_onnx:cpu` |
| moonshine | ttfp_ms | 81.0 | 73.0 | `moonshine:cuda` | `moonshine:cpu` |
| moonshine | request_ms | 3658.0 | 3389.0 | `moonshine:cuda` | `moonshine:cpu` |
| piper | ttfa_ms | 262.0 | 91.0 | `piper:cuda` | `piper:cpu` |
| piper | request_ms | 262.0 | 91.0 | `piper:cuda` | `piper:cpu` |
| kokoro | ttfa_ms | 1114.0 | 1119.0 | `kokoro:cuda` | `kokoro:cpu` |
| kokoro | request_ms | 1115.0 | 1119.0 | `kokoro:cuda` | `kokoro:cpu` |

## Warm

| engine | metric | CUDA | CPU | CUDA backend | CPU backend |
| --- | ---: | ---: | ---: | --- | --- |
| sherpa | ttfp_ms | 132.3 | 105.3 | `sherpa_onnx:cuda` | `sherpa_onnx:cpu` |
| sherpa | request_ms | 617.7 | 490.3 | `sherpa_onnx:cuda` | `sherpa_onnx:cpu` |
| moonshine | ttfp_ms | 64.7 | 63.0 | `moonshine:cuda` | `moonshine:cpu` |
| moonshine | request_ms | 3215.0 | 3445.0 | `moonshine:cuda` | `moonshine:cpu` |
| piper | ttfa_ms | 97.3 | 103.7 | `piper:cuda` | `piper:cpu` |
| piper | request_ms | 97.3 | 104.0 | `piper:cuda` | `piper:cpu` |
| kokoro | ttfa_ms | 1115.0 | 1095.0 | `kokoro:cuda` | `kokoro:cpu` |
| kokoro | request_ms | 1115.0 | 1095.3 | `kokoro:cuda` | `kokoro:cpu` |

These are smoke-cell timings on short audio/text. The main acceptance proof here is CUDA EP registration and fail-loud accelerator resolution under the static ORT policy; the small inputs do not consistently benefit from GPU execution.
