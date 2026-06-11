# piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default

- outcome: `blocked`
- reason: `artifact_unauthorized`
- acceptance: `blocked`
- failure_reason: `child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781132594513-3432241-0b8e8e5e-spark-2f6e-aarch64-cpu/logs/piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default.log`
- bundle: `piper_en_us_ljspeech_medium`
- capability: `tts`
- scenario: `tts_synthesis_smoke`
- profile: `local-cpu-aarch64`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cpu`
- resolved_accelerator: `cpu`
- backend: `ort`
- checkpoint_format: `onnx`
- quantization: `default`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `None`
- cargo_features: `model-piper-en-us-ljspeech-medium`
- accelerator_backend_mode: `cpu`
- accelerator_offload: `None`
- child_build_status: `101`
- child_build_duration_ms: `3796`

## Runtime Environment

- HF_TOKEN_PRESENT: `[REDACTED_PRESENT]`

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-aarch64 --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e
```

## Child Log Tail

```text
   Compiling ort v2.0.0-rc.12
   Compiling motlie-model-ort v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/ort)
   Compiling motlie-model-piper v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/piper)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/rustcKESJ91/symbols.o" "<242 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/{libtoml-8523e66f258a0630,libtoml_writer-5daa8c443d707fc1,libwinnow-85d655321ec0b38b,libtoml_parser-1fdceff1fcb84d5f,libwinnow-5bc763a7c6017c74,libserde_spanned-003fa5dd69123f88,libtoml_datetime-506d768c9eb166d7,libmotlie_eval_tools-bc7d55bf8157a5c1,libmotlie_voice-ae19892c7d85156d,libhound-234c7bde373ce109,libmotlie_models-f8813cfabc7ef702,libhf_hub-f4480a9c660141ce,libdirs-407a9f8f3a3c41dd,libdirs_sys-eb70da98475f3f15,liboption_ext-d52998b2c756ecfe,libnum_cpus-911b525cce4db20d,libureq-c1cc8ac92e6473e6,libwebpki_root_certs-e525f9e3b0a56a1f,libwebpki_roots-a450e715189f78d7,libder-40c01ffba7e45893,libpem_rfc7468-14f96d15b4aa90a2,libbase64ct-c109389ba5465ec4,librustls-1ddf2d390f9d01f3,libsubtle-35c28a651b9007aa,libwebpki-58cd3b4b7c1693c6,libring-9222b5f2adf4b06b,libgetrandom-90415ba81e8f14fb,libuntrusted-7edb9587b2358dc7,libsocks-a6fd8bc896017a2f,libbyteorder-8e3a52ca39823dc3,libflate2-2840effdd2760d5a,libminiz_oxide-9f44e6d1f40953c9,libsimd_adler32-10b9271f1afdef6b,libcrc32fast-09b093306eefc780,libutf8_zero-3981b5325a7ff3d2,libureq_proto-6af62399997dc7b3,libreqwest-aba131ce03241ada,libserde_urlencoded-8c22af4f9528e7a0,libryu-e5183732a1862270,librustls_pki_types-6da92f420379f0b5,libzeroize-abcadba4659cba69,libhyper_tls-4c5d972b702b42b1,libtokio_native_tls-1c5447b5cf65bf1b,libmime-21bd422aaab87a90,libencoding_rs-3e7e8fc25c565858,libtower_http-e6781189658ed38e,libiri_string-9ae110d7b00b3c80,libtower-e86deae034dae97d,libtower_layer-7085ad4e1462a70a,libnative_tls-5e89f014d756e0d0,libopenssl_probe-55d9138be76a271c,libopenssl-ce33b91dcb926d29,libbitflags-47b52d0babcffa4d,libforeign_types-047b4cd149152e99,libforeign_types_shared-f355e503206bb418,libopenssl_sys-237d020726f353bd,libhyper_util-314757c338544a94,libbase64-f35e2ca20aa812e6,libipnet-35433da69fabb362,libtower_service-33c8262433c445f4,libhyper-11510afe9d39eeab,libwant-21710d653b7b8fcd,libtry_lock-fbc28f258f7a9896,libhttparse-eef894b7f2851d06,libh2-09ec2233d490e057,libindexmap-ce5ea76b225072a6,libequivalent-33e6ad1a752de5f3,libhashbrown-29ce381f591c8aa7,libfnv-1060827e9741d772,libatomic_waker-01ce14e89d8fea5f,libtokio_util-678a2481059cd233,libtokio-60e373a176fdf915,libsignal_hook_registry-45179c7a68484839,liberrno-1e3b244ef39ee9d0,libsocket2-0b05831a64faf281,libmio-74ce859baba4ef9e,libparking_lot-ad6d1cb3dd4969fa,libparking_lot_core-c7e3d5de208a5ef4,liblock_api-ac7d916c1f8ac997,libscopeguard-106e9c94b54819d7,libhttp_body_util-f43788dc0308473e,libhttp_body-b8f16dc6cbab6b6e,liblog-05863d0ec06ec869,liburl-f66945d1c80989d3,libidna-9198cb18f0bc9deb,libidna_adapter-36636cb40f2928a5,libicu_properties-ba8e3f6f67563866,libicu_properties_data-7023304c1df6169c,libicu_normalizer-14a2bec5a8980a32,libicu_normalizer_data-54700c9ff37ecb74,libicu_collections-128440bd5a794800,libutf8_iter-0b703490ec99838d,libpotential_utf-1980d58c8cedb010,libicu_provider-7695cedcc199f349,libicu_locale_core-31fbe2e54e0a6a13,libtinystr-9d6e3d76e5144018,liblitemap-9bc427f8094a88b0,libwriteable-f11d7dd9b56b73dd,libzerovec-514a864a25a33124,libzerotrie-bb40d7988118e973,libyoke-b2a63506b788d830,libstable_deref_trait-39d9de628fe691db,libzerofrom-be8415be6a207edb,libform_urlencoded-54580b4da52f9b6d,libpercent_encoding-e47eb41ae2bc68cc,libhttp-4522b56cfb4d12a0,libbytes-4247db877434128f,libsync_wrapper-082041bf78244294,librand-8df4cf669f44002c,librand_chacha-aa01496b3b1a7120,libppv_lite86-bfdfa85cd99257d5,libzerocopy-10ced6711939226b,librand_core-d36324fd714c37c1,libgetrandom-948ceb86a0ac004b,libcfg_if-2e4f58b92fb84752,libfutures-411897fce9e42bb6,libfutures_executor-5fffbdf9ee76fee7,libfutures_util-5ad60c832b43203d,libfutures_io-d629d4286bbbe8c2,libslab-ccebb8a113dcd0c1,libfutures_channel-4cdfc161363e5205,libfutures_sink-06bf2e3e61cb09dc,libfutures_task-a0ea3c8d1c65f0f8,libfutures_core-a8d1908a820ed620,libindicatif-55d708d30917cfb9,libportable_atomic-202ff6bd85df128a,libunit_prefix-96471205f7bd70e3,libconsole-53c941f093c5f389,libunicode_width-18ae8f0ab45f63f2,libmotlie_model_piper-68a6c56e747477f8,libmotlie_model_ort-5fb3f219697041b3,libort-147fb2ac469a41aa,libort_sys-0806a18299f997eb,libndarray-82ac1c94486ddaeb,libmatrixmultiply-b7d374a32a363aa5,libnum_complex-6c8eb17e1087b6e8,libnum_integer-de486cd4cde16489,libnum_traits-d8567fb33f565621,librawpointer-c90cce43c1bfab56,libsmallvec-ba5bb2deffe9effc,libmotlie_model_espeak_ng-140f641a5b585dea,libmotlie_model-66c7511974124686,libtracing-032e0e31e5adf251,libpin_project_lite-666652f4637f721a,libtracing_core-e34111d09537a406,libonce_cell-084c6035348deddb,libschemars-c9f111e418c42a9c,libdyn_clone-ffbbdb565985a0ca,libserde-52507de9e9c8b2ee,libref_cast-3cee8f96d2987e9d,libserde_json-2a6177af187bf2e1,libitoa-16f8602d840f0338,libzmij-203b405024600965,libserde_core-7c5b515cb5b42ffb,libsysinfo-74e88aa27b56ef34,libmemchr-5934394692a4e9bc,liblibc-44afc8ed4ffe1f6a,libthiserror-aedaa0afa7058a92,libanyhow-51c249e542cb5d89}.rlib" "<sysroot>/lib/rustlib/aarch64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lssl" "-lcrypto" "-lespeak-ng" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/rustcKESJ91/raw-dylibs" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/ring-6417479215271a58/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/build/motlie-model-espeak-ng-30a538f136c94448/out/lib" "-L" "/usr/lib/aarch64-linux-gnu" "-L" "<sysroot>/lib/rustlib/aarch64-unknown-linux-gnu/lib" "-o" "/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/evals-07383608c1afe6de" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: /usr/bin/ld: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/libort-147fb2ac469a41aa.rlib(ort-147fb2ac469a41aa.ort.f34837289b1c93a2-cgu.07.rcgu.o): in function `ort::setup_api':
          /home/dchung/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-2.0.0-rc.12/src/lib.rs:207:(.text._ZN3ort9setup_api17h05a2e86c418849a1E+0xc): undefined reference to `OrtGetApiBase'
          /usr/bin/ld: /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/deps/libort_sys-0806a18299f997eb.rlib(ort_sys-0806a18299f997eb.ort_sys.777c219d3fa4edde-cgu.0.rcgu.o):/home/dchung/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ort-sys-2.0.0-rc.12/src/link_error.rs:15:(.data.rel.ro._ZN7ort_sys10link_error1X17ha16c0fc68aa89747E+0x0): undefined reference to `
          
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
