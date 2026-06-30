# whisper_base_en__asr_short_transcription__smoke__ggml_default

- outcome: `blocked`
- reason: `resource_gate_failed`
- acceptance: `blocked`
- failure_reason: `resources section not accepted: resource metric gpu_memory_peak_bytes blocked: CUDA peak VRAM sampler not instrumented`
- bundle: `whisper_base_en`
- capability: `asr`
- scenario: `asr_short_transcription`
- profile: `dgx-spark`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cuda`
- resolved_accelerator: `cuda`
- backend: `whisper_cpp`
- checkpoint_format: `ggml`
- quantization: `default`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `debug`
- cargo_features: `model-whisper-base-en, whisper-cpp-cuda, cuda`
- accelerator_backend_mode: `whisper_cpp:cuda`
- accelerator_offload: `cuda_execution_provider=on;selected_device=0`
- child_build_status: `0`
- child_build_duration_ms: `10717`

## Runtime Environment

- CUDA_VISIBLE_DEVICES: `None`
- MOTLIE_GGUF_BINDGEN_INCLUDE_WIRED: `None`
- MOTLIE_MODEL_FORCE_CPU: `None`
- MOTLIE_MODEL_GPU_LAYERS: `None`
- MOTLIE_PAGED_ATTN_CONTEXT: `None`

## Repro Command

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/evals run --bundle whisper_base_en --scenario asr_short_transcription --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id whisper_base_en__asr_short_transcription__smoke__ggml_default --depth smoke --checkpoint-format ggml --artifact-quantization default --model-family whisper --backend whisper_cpp --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/logs/whisper_base_en__asr_short_transcription__smoke__ggml_default.log --child-build-status 0 --child-build-duration-ms 10717 --quiet-backend-logs
```

## Child Log Tail

```text
   Compiling bindgen v0.72.1
   Compiling whisper-rs-sys v0.15.0
   Compiling whisper-rs v0.16.0
   Compiling motlie-model-whisper-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/whisper_cpp)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 10.62s
whisper_init_from_file_with_params_no_state: loading model from '/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache/models--ggerganov--whisper.cpp/snapshots/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GB10, compute capability 12.1, VMM: yes
whisper_init_with_params_no_state: devices    = 2
whisper_init_with_params_no_state: backends   = 2
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 2 (base)
whisper_model_load: adding 1607 extra tokens
whisper_model_load: n_langs       = 99
whisper_model_load:        CUDA0 total size =   147.37 MB
whisper_model_load: model size    =  147.37 MB
whisper_backend_init_gpu: device 0: CUDA0 (type: 1)
whisper_backend_init_gpu: found GPU device 0: CUDA0 (type: 1, cnt: 0)
whisper_backend_init_gpu: using CUDA0 backend
whisper_init_state: kv self size  =    6.29 MB
whisper_init_state: kv cross size =   18.87 MB
whisper_init_state: kv pad  size  =    3.15 MB
whisper_init_state: compute buffer (conv)   =   17.24 MB
whisper_init_state: compute buffer (encode) =   85.88 MB
whisper_init_state: compute buffer (cross)  =    4.66 MB
whisper_init_state: compute buffer (decode) =   97.29 MB

whisper_full_with_state: strategy = 0, decoding with 1 decoders, temperature = 0.00


whisper_full_with_state: prompt[0] = [_SOT_]


whisper_full_with_state: id =   0, decoder = 0, token =  50363, p =  0.966, ts =    [_BEG_],  0.966, result_len =    0 '[_BEG_]'
whisper_full_with_state: id =   1, decoder = 0, token =    632, p =  0.721, ts =        [?],  0.000, result_len =    0 ' It'
whisper_full_with_state: id =   2, decoder = 0, token =   3568, p =  0.957, ts =        [?],  0.013, result_len =    0 ' appears'
whisper_full_with_state: id =   3, decoder = 0, token =    326, p =  0.986, ts =        [?],  0.072, result_len =    0 ' that'
whisper_full_with_state: id =   4, decoder = 0, token =    262, p =  0.943, ts =        [?],  0.049, result_len =    0 ' the'
whisper_full_with_state: id =   5, decoder = 0, token =   3767, p =  0.997, ts =        [?],  0.008, result_len =    0 ' continued'
whisper_full_with_state: id =   6, decoder = 0, token =    779, p =  0.997, ts =        [?],  0.058, result_len =    0 ' use'
whisper_full_with_state: id =   7, decoder = 0, token =    286, p =  0.954, ts =        [?],  0.087, result_len =    0 ' of'
whisper_full_with_state: id =   8, decoder = 0, token =    262, p =  0.963, ts =  [_TT_100],  0.104, result_len =    0 ' the'
whisper_full_with_state: id =   9, decoder = 0, token =   7931, p =  0.882, ts =        [?],  0.007, result_len =    0 ' Iron'
whisper_full_with_state: id =  10, decoder = 0, token =   1869, p =  0.494, ts =        [?],  0.009, result_len =    0 ' Man'
whisper_full_with_state: id =  11, decoder = 0, token =   6050, p =  0.838, ts =        [?],  0.031, result_len =    0 ' suit'
whisper_full_with_state: id =  12, decoder = 0, token =    318, p =  0.844, ts =  [_TT_142],  0.126, result_len =    0 ' is'
whisper_full_with_state: id =  13, decoder = 0, token =  32253, p =  0.970, ts =  [_TT_149],  0.137, result_len =    0 ' accelerating'
whisper_full_with_state: id =  14, decoder = 0, token =    534, p =  0.956, ts =        [?],  0.085, result_len =    0 ' your'
whisper_full_with_state: id =  15, decoder = 0, token =   4006, p =  0.960, ts =  [_TT_194],  0.183, result_len =    0 ' condition'
whisper_full_with_state: id =  16, decoder = 0, token =     13, p =  0.893, ts =        [?],  0.052, result_len =    0 '.'
whisper_full_with_state: id =  17, decoder = 0, token =  50583, p =  0.042, ts =        [?],  0.042, result_len =   18 '[_TT_220]'
whisper_full_with_state: id =  18, decoder = 0, token =  50256, p =  0.985, ts =  [_TT_220],  0.683, result_len =   18 '<|endoftext|>'
whisper_full_with_state: decoder 0 completed
whisper_full_with_state: decoder  0: score = -0.28550, result_len =  18, avg_logprobs = -0.28550, entropy =  2.81336
whisper_full_with_state: best decoder = 0
single timestamp ending - skip entire chunk
seek = 498, seek_delta = 498
```
