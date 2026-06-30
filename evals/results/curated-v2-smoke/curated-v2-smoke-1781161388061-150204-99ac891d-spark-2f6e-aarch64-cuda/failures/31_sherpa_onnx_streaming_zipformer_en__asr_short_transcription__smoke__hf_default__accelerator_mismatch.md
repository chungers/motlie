# sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default

## Record

- schema_version: 3
- git_sha: 99ac891d8a2adabe823cce61b2a9fec0aa5dbde3
- run_id: curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda
- profile: dgx-spark
- host_id: spark-2f6e
- arch: aarch64
- capability: asr
- bundle_id: sherpa_onnx_streaming_zipformer_en
- backend: sherpa_onnx
- checkpoint_format: onnx
- quantization: default
- requested_accelerator: cuda
- resolved_accelerator: cpu
- accelerator_backend_mode: sherpa_onnx:cpu
- accelerator_use_proof_source: backend_observation
- outcome: blocked
- reason: accelerator_mismatch
- overall_status: blocked
- failure_reason: accelerator section not accepted: requested=cuda resolved=cpu reason=accelerator_mismatch
- child_build_profile: release
- child_build_status: 0

## Repro

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle sherpa_onnx_streaming_zipformer_en --scenario asr_short_transcription --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default --depth smoke --checkpoint-format onnx --artifact-quantization default --model-family sherpa_onnx --backend sherpa_onnx --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/logs/sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.log --child-build-status 0 --child-build-duration-ms 58633 --quiet-backend-logs
```

Child build command:

```sh
n/a
```

## Child Log Tail

```text
   Compiling zerofrom v0.1.8
   Compiling zeroize v1.8.2
   Compiling stable_deref_trait v1.2.1
   Compiling openssl-sys v0.9.116
   Compiling percent-encoding v2.3.2
   Compiling bzip2-sys v0.1.13+1.0.8
   Compiling foreign-types-shared v0.1.1
   Compiling writeable v0.6.3
   Compiling litemap v0.8.2
   Compiling base64 v0.22.1
   Compiling httparse v1.10.1
   Compiling yoke v0.8.3
   Compiling bytes v1.11.1
   Compiling base64ct v1.8.3
   Compiling native-tls v0.2.18
   Compiling utf8_iter v1.0.4
   Compiling rustls-pki-types v1.14.1
   Compiling foreign-types v0.3.2
   Compiling openssl v0.10.80
   Compiling xattr v1.6.1
   Compiling icu_properties_data v2.2.0
   Compiling icu_normalizer_data v2.2.0
   Compiling filetime v0.2.29
   Compiling matrixmultiply v0.3.10
   Compiling zerovec v0.11.6
   Compiling zerotrie v0.2.4
   Compiling byteorder v1.5.0
   Compiling openssl-probe v0.2.1
   Compiling ring v0.17.14
   Compiling utf8-zero v0.8.1
   Compiling pem-rfc7468 v1.0.0
   Compiling tar v0.4.46
   Compiling smallvec v1.15.1
   Compiling socks v0.3.4
   Compiling hmac-sha256 v1.1.14
   Compiling webpki-root-certs v1.0.7
   Compiling untrusted v0.9.0
   Compiling lzma-rust2 v0.15.8
   Compiling http v1.4.1
   Compiling crc32fast v1.5.0
   Compiling der v0.8.0
   Compiling simd-adler32 v0.3.9
   Compiling adler2 v2.0.1
   Compiling webpki-roots v1.0.7
   Compiling form_urlencoded v1.2.2
   Compiling once_cell v1.21.4
   Compiling tinystr v0.8.3
   Compiling potential_utf v0.1.5
   Compiling subtle v2.6.1
   Compiling webpki-roots v0.26.11
   Compiling miniz_oxide v0.8.9
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling icu_collections v2.2.0
   Compiling icu_locale_core v2.2.0
   Compiling flate2 v1.1.9
   Compiling bzip2 v0.4.4
   Compiling ureq-proto v0.6.0
   Compiling nalgebra v0.33.3
   Compiling ndarray v0.17.2
   Compiling icu_provider v2.2.0
   Compiling icu_normalizer v2.2.0
   Compiling icu_properties v2.2.0
   Compiling rustls v0.23.40
   Compiling idna_adapter v1.2.2
   Compiling idna v1.1.0
   Compiling ureq v3.3.0
   Compiling rustls-webpki v0.103.13
   Compiling url v2.5.8
   Compiling ort-sys v2.0.0-rc.12 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/third_party/ort-sys)
   Compiling ort v2.0.0-rc.12
   Compiling ureq v2.12.1
   Compiling motlie-model-ort v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/ort)
   Compiling sherpa-onnx-sys v1.13.2 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/vendor/sherpa-onnx-sys)
   Compiling statrs v0.18.0
   Compiling mistralrs-core v0.8.1
   Compiling sherpa-onnx v1.13.2
   Compiling motlie-model-sherpa-onnx v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/sherpa_onnx)
   Compiling mistralrs v0.8.1
   Compiling motlie-model-mistral v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/mistral)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `release` profile [optimized] target(s) in 58.55s

```
