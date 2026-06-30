# Failure: `piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default`

- bundle: `piper_en_us_ljspeech_medium`
- capability: `tts`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `ort`
- checkpoint_format: `onnx`
- quantization: `default`
- outcome: `blocked`
- reason: `feature_build_failed`
- child_build_status: `101`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-piper-en-us-ljspeech-medium
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling motlie-model-ort v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/ort)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/models)
   Compiling motlie-model-piper v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/piper)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/bins/evals)
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-m64" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/rustc6caxsB/symbols.o" "<243 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/{libtoml-cba3e9b0bb13f7c3,libtoml_writer-9b78e6b98a724d81,libwinnow-986d8e13b44c7141,libtoml_parser-b49874123ad9f866,libwinnow-86e1480d8c346384,libserde_spanned-13ec4bf454f6ee04,libtoml_datetime-081e4fd285a6af1b,libmotlie_eval_tools-af3acd25bcb92517,libmotlie_voice-43fabba3083b3c4e,libhound-804dd2ac132ada35,libmotlie_models-15572ba57eee5b72,libhf_hub-434ffc26bf2021f8,libdirs-ba705f9b8a2bfd36,libdirs_sys-12954f18e3e2ae66,liboption_ext-16a134393a8c90e7,libnum_cpus-ad083eb91c835f57,libureq-adb5178f034c76f1,libwebpki_root_certs-f6d650f4f9e33a23,libwebpki_roots-6691feb873e2b191,libder-d26411d44085b60a,libpem_rfc7468-41524f34ee02be05,libbase64ct-ff5a7268a7d077fa,librustls-095c1ac49ec9a68f,libsubtle-38f50f8cde93854a,libwebpki-3a941b64a306dcfe,libring-80f8767a2e559a43,libgetrandom-70951ca95edf8be0,libuntrusted-65f923ea859fb627,libsocks-43cb2d2f1659178d,libbyteorder-56459556ee3875a0,libflate2-5989e2e50df03ee7,libminiz_oxide-1d5934c921a6e760,libsimd_adler32-0fb34f6e23a790bf,libcrc32fast-b2a67b3a037cfcf4,libutf8_zero-12181c46dbbf822f,libureq_proto-ae96a17b08292b71,libreqwest-6e85f006f2b21a31,libserde_urlencoded-626f9430fec66678,libryu-4b90c1dcb676432e,librustls_pki_types-d1eb812402e5d25f,libzeroize-cafe2ada39f15181,libhyper_tls-43774de70e07b337,libtokio_native_tls-ee8e88f7797f29e6,libmime-9ed45721895d8c45,libencoding_rs-0912fa57cf973019,libtower_http-4b72b441f3bb06e2,libiri_string-ccbc29d9951208a5,libtower-75dc4a26dbe8e05d,libtower_layer-b9ce531dff4264ac,libnative_tls-3045a747d2529577,libopenssl_probe-c752c38aa92d8eb4,libopenssl-2384bb27302a56f3,libbitflags-7103a049596a7514,libforeign_types-a2bd76f48ebf9049,libforeign_types_shared-4680065068a92ee0,libopenssl_sys-99431d48f2521072,libhyper_util-b5cac3d4d0240357,libbase64-93d13499e98064b8,libipnet-6536f53fc1735499,libtower_service-4f70a3a8141e9637,libhyper-27b7096b81e6e476,libwant-1e88270185f26d8d,libtry_lock-25726098f143e8b6,libhttparse-6bdfc217418d6d1b,libh2-5af124d5ae43b48c,libindexmap-6c2f5d8bd6cae426,libequivalent-09a05a12e658fb17,libhashbrown-77e1cacfed5e9cdf,libfnv-ab3b3d0161207bc5,libatomic_waker-199214763a0024c7,libtokio_util-b6e260d52b3d7359,libtokio-af21671fd3c92184,libsignal_hook_registry-58d3bc5bc3e4bb83,liberrno-51d9838602b94dbf,libsocket2-1d865fb7e5be3ffc,libmio-6c84c5851adf537b,libparking_lot-4556e20c6bd84cce,libparking_lot_core-da9311fb5b991bf4,liblock_api-371ca4ea31f9ee70,libscopeguard-094b4676443ff474,libhttp_body_util-b3b46f5a70819a1d,libhttp_body-239fd26bbbf4b3b4,liblog-d7ffaa050103d639,liburl-58eaa1df08005ad6,libidna-6fd112e32b2fb13f,libidna_adapter-7519a7b731a0465c,libicu_properties-070b64c122bac2d5,libicu_properties_data-bc7a629dd0c180ce,libicu_normalizer-0e842b51dc51e20b,libicu_normalizer_data-e5ab749685dfccb2,libicu_collections-55713d73f765c936,libutf8_iter-a172792584edd8df,libpotential_utf-03db8e1d59c00d8a,libicu_provider-83cbf0f5960c535e,libicu_locale_core-a23d21773a133daa,libtinystr-95e153d9065de70f,liblitemap-76e700d5cb307359,libwriteable-677268dd1071aff5,libzerovec-b61409dd219eb9ed,libzerotrie-e254cdaf2b8dafce,libyoke-dcdaba2bd2852beb,libstable_deref_trait-ef45a612cfe6babc,libzerofrom-a1a2472de1e08c08,libform_urlencoded-03d793c84e04282e,libpercent_encoding-b1da3b8fc2e1026a,libhttp-12a5a172b047804c,libbytes-e030a54cc1f3d21f,libsync_wrapper-4af35ca15e8ea58c,librand-88f6157e4771468e,librand_chacha-e75f9ac4a053fa05,libppv_lite86-a3fdefe0818f4d8d,libzerocopy-3fb31ecfa03bfedb,librand_core-036d4942af85bfde,libgetrandom-8d91d45945263cac,libcfg_if-595cd1fd9b5b1165,libfutures-662d576140092b6a,libfutures_executor-f7b01d6789a6e124,libfutures_util-1a81f1a2ea8058ec,libfutures_io-4e166f08ca52ae97,libslab-ce0f3f0d172355ee,libfutures_channel-e5eae3c435cf39b0,libfutures_sink-bcea5761c467004c,libfutures_task-491ceb30f42251ea,libfutures_core-4009a2020ea13d93,libindicatif-9cc029e409d8f727,libportable_atomic-32032d9119346fe1,libunit_prefix-688a168f93c6d359,libconsole-5079d841105e207c,libunicode_width-a98708a0d0280085,libmotlie_model_piper-371b5d42e6c6fc92,libmotlie_model_ort-27b9ab20af966ba6,libort-7a6932cd01d5cf71,libort_sys-4fb63e18b05e291d,libndarray-1279af165970bb28,libmatrixmultiply-708a0b63627d65f3,libnum_complex-3cab5f2cfd9959f8,libnum_integer-e77b1a9660e26e13,libnum_traits-df03f96f611602b9,librawpointer-018099b4bc3fb67b,libsmallvec-0dfcb246392c0c7a,libmotlie_model_espeak_ng-88ce676648fbaaac,libmotlie_model-faf3bc242aabf0c0,libtracing-09e912dbf56106f0,libpin_project_lite-d7a7c9f9297e44b9,libtracing_core-f37f769a6d4fa6cf,libonce_cell-6fe0e84c103b3d0a,libschemars-c32024c7b8e1b4e0,libdyn_clone-8ea923e7b9221ec6,libserde-ecdc0a9676c398e4,libref_cast-7b23e95790cbaff9,libserde_json-930527456eafe27e,libitoa-d62e748016f8bd79,libzmij-0e702112b516ef8a,libserde_core-bdf0eb8a40e58e47,libsysinfo-29f2005bd2eab19e,libmemchr-1ba191a0e30eb7bf,liblibc-b1f611c28b7e9ae9,libthiserror-c9b33271ff70b79c,libanyhow-81e0f801a5d79612}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,libcfg_if-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,liblibc-*,librustc_std_workspace_core-*,liballoc-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lssl" "-lcrypto" "-lespeak-ng" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/rustc6caxsB/raw-dylibs" "-B<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld" "-fuse-ld=lld" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/build/ring-c427ff05c0e92af0/out" "-L" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/build/motlie-model-espeak-ng-e6b26efa6e4073c0/out/lib" "-L" "/usr/lib/x86_64-linux-gnu" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/evals-79b3072452a7e586" "-Wl,--gc-sections" "-pie" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: rust-lld: error: undefined symbol: OrtGetApiBase
          >>> referenced by lib.rs:207 (src/lib.rs:207)
          >>>               ort-7a6932cd01d5cf71.ort.fe5b0ce3a4fd5b5a-cgu.13.rcgu.o:(ort::setup_api::hdcf5855f72020e8f) in archive /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/libort-7a6932cd01d5cf71.rlib

          rust-lld: error: undefined symbol:

          The ort-sys crate could not link to ONNX Runtime because:
              - `libonnxruntime` is not configured via `pkg-config`
              - ort-sys was instructed not to download prebuilt binaries (`cargo build --offline`), or the `download-binaries` feature is not enabled
              - Neither `ORT_LIB_PATH` or `ORT_IOS_XCFWK_PATH` (for iOS) were set to link to custom binaries

          To rectify this:
              - Compile ONNX Runtime from source and manually configure linking (see https://ort.pyke.io/setup/linking for more information)
              - Enable the `download-binaries` feature if the target is supported
              - Enable ort's `alternative-backend` feature if you intend to use a different backend (or ort-sys' `disable-linking` feature if you use this crate directly)

          >>> referenced by link_error.rs:15 (src/link_error.rs:15)
          >>>               ort_sys-4fb63e18b05e291d.ort_sys.c416e39899edfbb3-cgu.0.rcgu.o:(ort_sys::link_error::X::h45e0e5f93e2a9f85) in archive /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/deps/libort_sys-4fb63e18b05e291d.rlib
          collect2: error: ld returned 1 exit status


error: could not compile `evals` (bin "evals") due to 1 previous error

```
