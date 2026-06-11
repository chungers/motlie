# moonshine_streaming_en__asr_short_transcription__smoke__hf_default

## Record

- schema_version: 3
- git_sha: e14eaa7e00ba51c4e35636f9312da5e5fb963238
- run_id: curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda
- profile: dgx-spark
- host_id: spark-2f6e
- arch: aarch64
- capability: asr
- bundle_id: moonshine_streaming_en
- backend: ort
- checkpoint_format: onnx
- quantization: default
- requested_accelerator: cuda
- resolved_accelerator: cpu
- accelerator_backend_mode: moonshine:cpu
- accelerator_use_proof_source: backend_observation
- outcome: blocked
- reason: accelerator_mismatch
- overall_status: blocked
- failure_reason: accelerator section not accepted: requested=cuda resolved=cpu reason=accelerator_mismatch
- child_build_profile: release
- child_build_status: 0

## Repro

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/target/release/evals run --bundle moonshine_streaming_en --scenario asr_short_transcription --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-targeted-e14eaa7e/curated-v2-smoke/curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id moonshine_streaming_en__asr_short_transcription__smoke__hf_default --depth smoke --checkpoint-format onnx --artifact-quantization default --model-family moonshine --backend ort --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-targeted-e14eaa7e/curated-v2-smoke/curated-v2-smoke-1781168781390-337905-e14eaa7e-spark-2f6e-aarch64-cuda/logs/moonshine_streaming_en__asr_short_transcription__smoke__hf_default.log --child-build-status 0 --child-build-duration-ms 16278 --quiet-backend-logs
```

Child build command:

```sh
n/a
```

## Child Log Tail

```text
   Compiling serde_core v1.0.228
   Compiling stable_deref_trait v1.2.1
   Compiling libc v0.2.186
   Compiling cc v1.2.63
   Compiling tracing-core v0.1.36
   Compiling serde_json v1.0.150
   Compiling bitflags v2.13.0
   Compiling num-traits v0.2.19
   Compiling httparse v1.10.1
   Compiling zeroize v1.8.2
   Compiling foreign-types-shared v0.1.1
   Compiling base64ct v1.8.3
   Compiling bytes v1.11.1
   Compiling native-tls v0.2.18
   Compiling getrandom v0.3.4
   Compiling zerofrom v0.1.8
   Compiling foreign-types v0.3.2
   Compiling zerocopy v0.8.50
   Compiling futures-util v0.3.32
   Compiling rustls-pki-types v1.14.1
   Compiling base64 v0.22.1
   Compiling openssl-probe v0.2.1
   Compiling byteorder v1.5.0
   Compiling xattr v1.6.1
   Compiling yoke v0.8.3
   Compiling filetime v0.2.29
   Compiling utf8-zero v0.8.1
   Compiling tracing v0.1.44
   Compiling pem-rfc7468 v1.0.0
   Compiling percent-encoding v2.3.2
   Compiling zerovec v0.11.6
   Compiling zerotrie v0.2.4
   Compiling http v1.4.1
   Compiling socks v0.3.4
   Compiling tar v0.4.46
   Compiling webpki-root-certs v1.0.7
   Compiling der v0.8.0
   Compiling indexmap v2.14.0
   Compiling matrixmultiply v0.3.10
   Compiling hmac-sha256 v1.1.14
   Compiling lzma-rust2 v0.15.8
   Compiling regex-automata v0.4.14
   Compiling either v1.16.0
   Compiling parking_lot_core v0.9.12
   Compiling errno v0.3.14
   Compiling mio v1.2.1
   Compiling socket2 v0.6.4
   Compiling tinystr v0.8.3
   Compiling signal-hook-registry v1.4.8
   Compiling parking_lot v0.12.5
   Compiling potential_utf v0.1.5
   Compiling ureq-proto v0.6.0
   Compiling icu_locale_core v2.2.0
   Compiling icu_collections v2.2.0
   Compiling tokio v1.52.3
   Compiling num-integer v0.1.46
   Compiling rand_core v0.9.5
   Compiling num-complex v0.4.6
   Compiling getrandom v0.2.17
   Compiling sysinfo v0.36.1
   Compiling simd-adler32 v0.3.9
   Compiling openssl-sys v0.9.116
   Compiling bzip2-sys v0.1.13+1.0.8
   Compiling ring v0.17.14
   Compiling onig_sys v69.9.3
   Compiling esaxx-rs v0.1.10
   Compiling serde v1.0.228
   Compiling icu_provider v2.2.0
   Compiling rustls v0.23.40
   Compiling ndarray v0.17.2
   Compiling miniz_oxide v0.8.9
   Compiling primal-check v0.3.4
   Compiling icu_properties v2.2.0
   Compiling openssl v0.10.80
   Compiling icu_normalizer v2.2.0
   Compiling transpose v0.2.3
   Compiling rayon v1.12.0
   Compiling itertools v0.14.0
   Compiling schemars v1.2.1
   Compiling bzip2 v0.4.4
   Compiling console v0.15.11
   Compiling idna_adapter v1.2.2
   Compiling regex v1.12.3
   Compiling idna v1.1.0
   Compiling ppv-lite86 v0.2.21
   Compiling rand_chacha v0.9.0
   Compiling motlie-model v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/model)
   Compiling rand v0.9.4
   Compiling url v2.5.8
   Compiling is-terminal v0.4.17
   Compiling termcolor v1.4.1
   Compiling humantime v2.3.0
   Compiling transcribe-rs v0.3.11
   Compiling spm_precompiled v0.1.4
   Compiling dirs-sys v0.5.0
   Compiling indicatif v0.17.11
   Compiling env_logger v0.10.2
   Compiling ureq v3.3.0
   Compiling compact_str v0.9.1
   Compiling tokio-util v0.7.18
   Compiling tower v0.5.3
   Compiling monostate v0.1.18
   Compiling tokio-native-tls v0.3.1
   Compiling h2 v0.4.14
   Compiling tower-http v0.6.11
   Compiling rayon-cond v0.4.0
   Compiling ahash v0.8.12
   Compiling flate2 v1.1.9
   Compiling rustfft v6.4.1
   Compiling serde_urlencoded v0.7.1
   Compiling dary_heap v0.3.9
   Compiling futures-executor v0.3.32
   Compiling console v0.16.3
   Compiling dirs v6.0.0
   Compiling futures v0.3.32
   Compiling num_cpus v1.17.0
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/models)
   Compiling motlie-voice v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/voice)
   Compiling serde_spanned v1.1.1
   Compiling toml_datetime v0.7.5+spec-1.1.0
   Compiling motlie-eval-tools v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/eval-tools)
   Compiling ort-sys v2.0.0-rc.12 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/third_party/ort-sys)
   Compiling indicatif v0.18.4
   Compiling toml v0.9.12+spec-1.1.0
   Compiling hyper v1.10.1
   Compiling ort v2.0.0-rc.12
   Compiling rustls-webpki v0.103.13
   Compiling hyper-util v0.1.20
   Compiling hyper-tls v0.6.0
   Compiling reqwest v0.12.28
   Compiling onig v6.5.3
   Compiling tokenizers v0.21.4
   Compiling hf-hub v0.5.0
   Compiling motlie-model-moonshine v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/model/backends/moonshine)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/bins/evals)
    Finished `release` profile [optimized] target(s) in 16.21s

```
