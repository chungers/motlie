# sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default

- outcome: `blocked`
- reason: `feature_build_failed`
- acceptance: `blocked`
- failure_reason: `child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781131781834-3412429-0b8e8e5e-spark-2f6e-aarch64-cuda/logs/sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.log`
- bundle: `sherpa_onnx_streaming_zipformer_en`
- capability: `asr`
- scenario: `asr_short_transcription`
- profile: `dgx-spark`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cuda`
- resolved_accelerator: `cuda`
- backend: `sherpa_onnx`
- checkpoint_format: `onnx`
- quantization: `default`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `None`
- cargo_features: `model-sherpa-onnx-streaming, cuda, sherpa-onnx-cuda`
- accelerator_backend_mode: `backend_offload_unverified`
- accelerator_offload: `None`
- child_build_status: `101`
- child_build_duration_ms: `3577`

## Runtime Environment

- HF_TOKEN_PRESENT: `[REDACTED_PRESENT]`

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile dgx-spark --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e
```

## Child Log Tail

```text
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/rustcGLcXM0/symbols.o" "<240 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/{libtoml-8a195b6491f4908e,libtoml_writer-5daa8c443d707fc1,libwinnow-85d655321ec0b38b,libtoml_parser-1fdceff1fcb84d5f,libwinnow-5bc763a7c6017c74,libserde_spanned-e5952315555e960f,libtoml_datetime-340df028e5ef9b23,libmotlie_eval_tools-6e7d43a1e82ed9b4,libmotlie_voice-5f9376f551e71b4a,libhound-234c7bde373ce109,libmotlie_models-b6814eb9f81740c6,libhf_hub-dfcfc2cb06cef249,libdirs-bd428d4bdc550982,libdirs_sys-7387d64c47122e8d,liboption_ext-d52998b2c756ecfe,libnum_cpus-bf295b0cd0b7efcd,libureq-4417d451b8305a65,libwebpki_root_certs-e525f9e3b0a56a1f,libder-40c01ffba7e45893,libpem_rfc7468-14f96d15b4aa90a2,libbase64ct-c109389ba5465ec4,libsocks-d36035e564b63eb8,libbyteorder-8e3a52ca39823dc3,libflate2-a0240314127e9195,libminiz_oxide-e2d3579a255e2eab,libsimd_adler32-b30747c01c164b41,libcrc32fast-09b093306eefc780,libutf8_zero-3981b5325a7ff3d2,libureq_proto-53db891fd9e490d4,libreqwest-ae0f97a0f92e98af,libhyper_rustls-bb719d562b39303a,libwebpki_roots-a450e715189f78d7,libhyper_tls-323795732a1b9bf8,libtokio_native_tls-182241e0689dd304,libtokio_rustls-bc8f91e9bed064c3,librustls-6ef399d79464aad3,libsubtle-35c28a651b9007aa,libwebpki-c94b03e7d8df32ba,libaws_lc_rs-05b56ab398c9f9ae,libaws_lc_sys-b5f0ac58d6b88f48,libring-467296c32459bae4,libuntrusted-7edb9587b2358dc7,librustls_pki_types-6da92f420379f0b5,libzeroize-abcadba4659cba69,libserde_urlencoded-ec2c86ebaaa6ffd5,libryu-e5183732a1862270,libencoding_rs-3e7e8fc25c565858,libmime_guess-4ce76be3716e00a4,libunicase-0505b6a6e5c1fec8,libmime-21bd422aaab87a90,libtower_http-72b7e227deace386,libiri_string-9ae110d7b00b3c80,libtower-e037346748d3ecb7,libtower_layer-7085ad4e1462a70a,libnative_tls-64c06354686af91b,libopenssl_probe-55d9138be76a271c,libopenssl-887e8b05ba0cf307,libbitflags-19848f4e0aa402e6,libforeign_types-047b4cd149152e99,libforeign_types_shared-f355e503206bb418,libopenssl_sys-9d355ab28b54b355,libhyper_util-a55f94f935c5ab55,libbase64-f35e2ca20aa812e6,libipnet-35433da69fabb362,libtower_service-33c8262433c445f4,libhyper-541ff8ff54281bb3,libwant-21710d653b7b8fcd,libtry_lock-fbc28f258f7a9896,libhttparse-eef894b7f2851d06,libh2-500aa03404ef10db,libfnv-1060827e9741d772,libatomic_waker-01ce14e89d8fea5f,libtokio_util-006582bfbd74988c,libtokio-3ffdb39fb95aa29d,libsignal_hook_registry-20850aae29da7337,liberrno-8e18b150ecff3832,libsocket2-d097baa59500f1a4,libmio-1ced76643fcd66a1,libparking_lot-6c400d77df09c7be,libparking_lot_core-6513776c217cb376,liblock_api-ac7d916c1f8ac997,libscopeguard-106e9c94b54819d7,libhttp_body_util-f43788dc0308473e,libhttp_body-b8f16dc6cbab6b6e,liblog-1e296e7ee3c2cbab,liburl-0f0a8f6b1c9c6bbb,libidna-255e9d303ae863d4,libidna_adapter-ca666ad9b074d32e,libicu_properties-677a53d1bd3be5b3,libicu_properties_data-7023304c1df6169c,libicu_normalizer-c8ef8d229ba9ae89,libicu_normalizer_data-54700c9ff37ecb74,libicu_collections-f0d9b61ccd3ee2e2,libutf8_iter-0b703490ec99838d,libpotential_utf-16ff861096319fcc,libicu_provider-a51e3985acc40da9,libicu_locale_core-834f441c27abf400,libtinystr-edb9e7ca20c681a1,liblitemap-9bc427f8094a88b0,libwriteable-f11d7dd9b56b73dd,libzerovec-99f56e1e98f0c43a,libzerotrie-d503d2bb231e6f3f,libyoke-1595f2e207195d0b,libstable_deref_trait-f443daa990593162,libzerofrom-95b951523ce56e68,libform_urlencoded-54580b4da52f9b6d,libpercent_encoding-e47eb41ae2bc68cc,libhttp-4522b56cfb4d12a0,libbytes-4247db877434128f,libsync_wrapper-082041bf78244294,librand-e3533d9eef94c350,librand_chacha-b294a59f98d60c48,librand_core-c5cfdc54be1a3335,libgetrandom-96180e42f6d14f39,libfutures-95e03390969dc2b4,libfutures_executor-b9a30e6515ee62f5,libfutures_util-e414035f3c43f24f,libfutures_io-d629d4286bbbe8c2,libslab-ccebb8a113dcd0c1,libfutures_channel-4cdfc161363e5205,libfutures_sink-06bf2e3e61cb09dc,libfutures_task-a0ea3c8d1c65f0f8,libfutures_core-a8d1908a820ed620,libindicatif-bb4dc6a17929d674,libportable_atomic-202ff6bd85df128a,librayon-d9a1238905fcd62e,librayon_core-549e0cc7e2ef7463,libcrossbeam_deque-06e1cd62ed4e2fdc,libcrossbeam_epoch-adb2fcc539ce0d37,libcrossbeam_utils-90ee8190b5b0ce40,libeither-0c45825410349850,libunit_prefix-96471205f7bd70e3,libconsole-d354abfcbbaa8be3,libunicode_width-18ae8f0ab45f63f2,libmotlie_model_sherpa_onnx-7a1c2247cc0fd6ed,libmotlie_model_ort-a79008211941e9b4,libort-71d64a2f75134c88,libort_sys-8330e2af7a4b6d60,libndarray-c78b610c36d28413,libmatrixmultiply-2cb89866ffbbef16,librawpointer-c90cce43c1bfab56,libsmallvec-ba5bb2deffe9effc,libkaldi_native_fbank-79b15aad9dc2e0bd,librand-b238f71eccaebd5d,librand_chacha-4876037633b2f3d8,libppv_lite86-98245d5c2e8ae23a,libzerocopy-141ae66ebc91d8ea,librand_core-3ff96b8bcb83dd0b,libgetrandom-1a7b62e22f13f491,libcfg_if-2e4f58b92fb84752,librealfft-43c2ee1b94739312,librustfft-8a83bff64215da24,libprimal_check-170912f30d28b5ce,libtranspose-8fee9a99ed533e39,libstrength_reduce-ba9756bf89cee3df,libnum_integer-b13622e0f89c553f,libnum_complex-8760bb2033d7162f,libbytemuck-32d3786a1720e332,libnum_traits-ce3ad9bb2c2919fe,libmotlie_model-7898e1cba71e9d88,libtracing-42097c4c7aed133a,libpin_project_lite-666652f4637f721a,libtracing_core-8d36b6e0091e8cd5,libonce_cell-084c6035348deddb,libschemars-b3f5b4459020a20d,libdyn_clone-ffbbdb565985a0ca,libserde-d2b48fcde5a3646b,libref_cast-e97c41e6666803a8,libserde_json-854e07ecaf7cf624,libitoa-16f8602d840f0338,libzmij-203b405024600965,libindexmap-50633e12e2564b5c,libequivalent-33e6ad1a752de5f3,libhashbrown-29ce381f591c8aa7,libserde_core-97137f21a8a5b061,libsysinfo-b407eeebf5770678,libmemchr-5934394692a4e9bc,liblibc-aa0cb68a4b856ed4,libthiserror-7be95ae7c4af7b2d,libanyhow-51c249e542cb5d89}.rlib" "<sysroot>/lib/rustlib/aarch64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lssl" "-lcrypto" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/rustcGLcXM0/raw-dylibs" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/aws-lc-sys-1f809d5571de78bb/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/ring-e07d29b98beb0f82/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/candle-kernels-07a36fd9f1ed5bca/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/onig_sys-5018f4b07aa6563f/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/mistralrs-core-a21cadeeadc943eb/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/mistralrs-paged-attn-92ed6d745620905e/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/mistralrs-quant-c3f052ceb673762f/out" "-L" "/usr/lib" "-L" "/usr/local/cuda/lib64" "-L" "/usr/local/cuda/lib64/stubs" "-L" "<sysroot>/lib/rustlib/aarch64-unknown-linux-gnu/lib" "-o" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/evals-7b1d16bbb4bafe01" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: /usr/bin/ld: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/libort-71d64a2f75134c88.rlib(ort-71d64a2f75134c88.ort.9808d5c3cfd9c436-cgu.06.rcgu.o): in function `ort::setup_api':
          /home/dchung/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.12/src/lib.rs:207:(.text._ZN3ort9setup_api17h976b91134d853e42E+0xc): undefined reference to `OrtGetApiBase'
          /usr/bin/ld: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/libort_sys-8330e2af7a4b6d60.rlib(ort_sys-8330e2af7a4b6d60.ort_sys.9dc5b3aa127774a4-cgu.0.rcgu.o):/home/dchung/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-sys-2.0.0-rc.12/src/link_error.rs:15:(.data.rel.ro._ZN7ort_sys10link_error1X17h56dcf450084348cbE+0x0): undefined reference to `
          
          The ort-sys crate could not link to ONNX Runtime because:
          	- `libonnxruntime` is not configured via `pkg-config`
          	- ort-sys was instructed not to download prebuilt binaries (`cargo build --offline`), or the `download-binaries` feature is not enabled
          	- Neither `ORT_LIB_PATH` or `ORT_IOS_XCFWK_PATH` (for iOS) were set to link to custom binaries
          
          To rectify this:
          	- Compile ONNX Runtime from source and manually configure linking (see https://ort.pyke.io/setup/linking for more information)
          	- Enable the `download-binaries` feature if the target is supported
          	- Enable ort's `alternative-backend` feature if you intend to use a different backend (or ort-sys' `disable-linking` feature if you use this crate directly)
          '
          collect2: error: ld returned 1 exit status
          
  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-lib)

error: could not compile `evals` (bin "evals") due to 1 previous error
```
