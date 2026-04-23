use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let vendor_dir = PathBuf::from("vendor/qwen3-tts.cpp");
    let api_cpp = vendor_dir.join("src/qwen3tts_c_api.cpp");
    let api_h = vendor_dir.join("src/qwen3tts_c_api.h");
    let ggml_dir = vendor_dir.join("ggml");
    let ggml_cmake = ggml_dir.join("CMakeLists.txt");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.gitmodules");
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join(".git").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("CMakeLists.txt").display()
    );
    println!("cargo:rerun-if-changed={}", api_cpp.display());
    println!("cargo:rerun-if-changed={}", api_h.display());
    println!("cargo:rerun-if-changed={}", ggml_cmake.display());

    validate_submodule_checkout(&vendor_dir, &api_cpp, &api_h, &ggml_cmake);
    build_ggml_submodule(&ggml_dir);

    let wrapper_dir = generate_wrapper_project(&vendor_dir);
    let mut config = cmake::Config::new(&wrapper_dir);
    config.profile("Release");
    config.build_target("qwen3tts_shared");
    config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define(
        "MOTLIE_VENDOR_DIR",
        canonicalize_existing_path(vendor_dir.clone()),
    );
    config.define(
        "MOTLIE_ENABLE_CUDA",
        if cfg!(feature = "cuda") { "ON" } else { "OFF" },
    );

    config.define(
        "MOTLIE_ENABLE_COREML",
        if cfg!(target_os = "macos") { "ON" } else { "OFF" },
    );
    if cfg!(target_os = "macos") {
        config.define("GGML_METAL", "ON");
    }
    if cfg!(target_env = "gnu") {
        config.define("CMAKE_EXE_LINKER_FLAGS", "-fopenmp");
        let mut shared_linker_flags = String::from("-fopenmp");
        if cfg!(target_os = "linux") {
            // Keep qwen3-tts.cpp's bundled ggml symbols local to libqwen3tts so
            // co-linked backends like whisper.cpp do not interpose them.
            shared_linker_flags.push_str(" -Wl,-Bsymbolic");
        }
        config.define("CMAKE_SHARED_LINKER_FLAGS", shared_linker_flags);
    }

    let dst = config.build();
    let lib_dir = locate_built_library_dir(&dst);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=qwen3tts");
    println!("cargo:libdir={}", lib_dir.display());
    if cfg!(feature = "cuda") {
        emit_cuda_link_directives();
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if cfg!(target_env = "gnu") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        if cfg!(target_env = "gnu") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    if let Ok(target) = env::var("TARGET") {
        println!("cargo:warning=qwen3-tts.cpp built for target {target}");
    }
    if cfg!(feature = "cuda") {
        println!(
            "cargo:warning=qwen3-tts.cpp CUDA build enabled via motlie wrapper; ggml-cuda is linked explicitly and upstream test targets are excluded from install builds"
        );
    }
}

fn validate_submodule_checkout(vendor_dir: &Path, api_cpp: &Path, api_h: &Path, ggml_cmake: &Path) {
    let init_hint = "run `git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp`";

    if !vendor_dir.exists() {
        panic!(
            "qwen3-tts.cpp submodule is missing at `{}`; {}",
            vendor_dir.display(),
            init_hint
        );
    }

    for required in [
        vendor_dir.join("CMakeLists.txt"),
        api_cpp.to_path_buf(),
        api_h.to_path_buf(),
    ] {
        if !required.is_file() {
            panic!(
                "qwen3-tts.cpp submodule checkout at `{}` is incomplete; missing `{}`; {}",
                vendor_dir.display(),
                required.display(),
                init_hint
            );
        }
    }

    if !ggml_cmake.is_file() {
        panic!(
            "qwen3-tts.cpp nested `ggml` submodule is missing at `{}`; {}",
            ggml_cmake.display(),
            init_hint
        );
    }
}

fn build_ggml_submodule(ggml_dir: &Path) {
    let build_dir = ggml_dir.join("build");
    let mut configure = Command::new("cmake");
    configure.arg("-S").arg(ggml_dir);
    configure.arg("-B").arg(&build_dir);
    configure.arg("-DCMAKE_BUILD_TYPE=Release");
    configure.arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");
    configure.arg("-DBUILD_SHARED_LIBS=OFF");
    configure.arg("-DGGML_STATIC=ON");
    configure.arg("-DGGML_BUILD_TESTS=OFF");
    configure.arg("-DGGML_BUILD_EXAMPLES=OFF");
    configure.arg(format!(
        "-DGGML_CUDA={}",
        if cfg!(feature = "cuda") { "ON" } else { "OFF" }
    ));
    if cfg!(target_os = "macos") {
        configure.arg("-DGGML_METAL=ON");
    }
    run_or_panic(&mut configure, "configure nested ggml submodule");

    let mut build = Command::new("cmake");
    build.arg("--build").arg(&build_dir);
    build.arg("--config").arg("Release");
    run_or_panic(&mut build, "build nested ggml submodule");

    if cfg!(feature = "cuda") {
        let ggml_cuda_archive = build_dir.join("src/ggml-cuda/libggml-cuda.a");
        if !ggml_cuda_archive.is_file() {
            panic!(
                "CUDA build requested but nested ggml did not produce `{}`; qwen3-tts.cpp CUDA linkage cannot proceed",
                ggml_cuda_archive.display()
            );
        }
    }
}

fn generate_wrapper_project(vendor_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let wrapper_dir = out_dir.join("cmake-wrapper");
    fs::create_dir_all(&wrapper_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create qwen3-tts.cpp wrapper directory `{}`: {error}",
            wrapper_dir.display()
        )
    });

    let wrapper_cmake = wrapper_dir.join("CMakeLists.txt");
    fs::write(&wrapper_cmake, render_wrapper_cmakelists(vendor_dir)).unwrap_or_else(|error| {
        panic!(
            "failed to write qwen3-tts.cpp wrapper CMakeLists `{}`: {error}",
            wrapper_cmake.display()
        )
    });
    println!("cargo:rerun-if-changed={}", wrapper_cmake.display());

    wrapper_dir
}

fn render_wrapper_cmakelists(_vendor_dir: &Path) -> String {
    r#"cmake_minimum_required(VERSION 3.14)
project(motlie-qwen3-tts-wrapper LANGUAGES CXX)

if(NOT DEFINED MOTLIE_VENDOR_DIR)
    message(FATAL_ERROR "MOTLIE_VENDOR_DIR must point at the qwen3-tts.cpp checkout")
endif()

if(NOT DEFINED MOTLIE_ENABLE_CUDA)
    set(MOTLIE_ENABLE_CUDA OFF)
endif()

if(NOT DEFINED MOTLIE_ENABLE_COREML)
    set(MOTLIE_ENABLE_COREML OFF)
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(QWEN3_TTS_COREML ${MOTLIE_ENABLE_COREML} CACHE BOOL "" FORCE)

add_subdirectory("${MOTLIE_VENDOR_DIR}" vendor-build)

foreach(test_target test_tokenizer test_encoder test_transformer test_decoder)
    if(TARGET ${test_target})
        set_target_properties(
            ${test_target}
            PROPERTIES
            EXCLUDE_FROM_ALL TRUE
            EXCLUDE_FROM_DEFAULT_BUILD TRUE
        )
    endif()
endforeach()

if(TARGET qwen3-tts-cli)
    set_target_properties(
        qwen3-tts-cli
        PROPERTIES
        EXCLUDE_FROM_ALL TRUE
        EXCLUDE_FROM_DEFAULT_BUILD TRUE
    )
endif()

if(MOTLIE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(MOTLIE_GGML_CUDA_LIB "${MOTLIE_VENDOR_DIR}/ggml/build/src/ggml-cuda/libggml-cuda.a")
    if(NOT EXISTS "${MOTLIE_GGML_CUDA_LIB}")
        message(FATAL_ERROR "Expected ggml CUDA archive not found at ${MOTLIE_GGML_CUDA_LIB}")
    endif()

    foreach(lib_target text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder)
        if(TARGET ${lib_target})
            target_link_libraries(${lib_target} PUBLIC "${MOTLIE_GGML_CUDA_LIB}")
        endif()
    endforeach()

    foreach(bin_target qwen3_tts qwen3tts_shared qwen3-tts-cli)
        if(TARGET ${bin_target})
            target_link_libraries(
                ${bin_target}
                PRIVATE
                "${MOTLIE_GGML_CUDA_LIB}"
                CUDA::cudart
                CUDA::cuda_driver
            )
        endif()
    endforeach()
endif()
"#
    .to_owned()
}

fn canonicalize_existing_path(path: PathBuf) -> PathBuf {
    fs::canonicalize(&path)
        .unwrap_or_else(|error| panic!("failed to canonicalize `{}`: {error}", path.display()))
}

fn locate_built_library_dir(dst: &Path) -> PathBuf {
    let candidates = [
        dst.join("build/vendor-build"),
        dst.join("build"),
        dst.join("vendor-build"),
        dst.join("lib"),
    ];

    for candidate in candidates {
        if candidate.is_dir() {
            let canonical = canonicalize_existing_path(candidate);
            if has_expected_library(&canonical) {
                return canonical;
            }
        }
    }

    panic!(
        "qwen3-tts.cpp build completed but no expected shared library was found under `{}`",
        dst.display()
    );
}

fn has_expected_library(lib_dir: &Path) -> bool {
    let candidate_names: &[&str] = if cfg!(target_os = "macos") {
        &["libqwen3tts.dylib", "libqwen3tts.0.dylib"]
    } else if cfg!(target_os = "windows") {
        &["qwen3tts.dll"]
    } else {
        &["libqwen3tts.so", "libqwen3tts.so.0"]
    };

    candidate_names
        .iter()
        .any(|name| lib_dir.join(name).is_file())
}

fn run_or_panic(command: &mut Command, action: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {action}: {error}"));
    if !status.success() {
        panic!("failed to {action}: exit status {status}");
    }
}

fn discover_cuda_lib_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    for key in ["CUDA_LIB_PATH", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(value) = env::var(key) {
            let root = PathBuf::from(value);
            dirs.push(root.clone());
            dirs.push(root.join("lib64"));
            dirs.push(root.join("targets/sbsa-linux/lib"));
            dirs.push(root.join("targets/sbsa-linux/lib/stubs"));
            dirs.push(root.join("targets/x86_64-linux/lib"));
            dirs.push(root.join("targets/x86_64-linux/lib/stubs"));
        }
    }

    dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/sbsa-linux/lib"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/sbsa-linux/lib/stubs"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/x86_64-linux/lib"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/x86_64-linux/lib/stubs"));
    dirs.push(PathBuf::from("/usr/local/cuda-13.0/lib64"));
    dirs.push(PathBuf::from("/usr/local/cuda-13.0/targets/sbsa-linux/lib"));
    dirs.push(PathBuf::from("/usr/local/cuda-13.0/targets/sbsa-linux/lib/stubs"));
    dirs.push(PathBuf::from("/usr/local/cuda-13.0/targets/x86_64-linux/lib"));
    dirs.push(PathBuf::from("/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs"));
    dirs.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));

    dirs.into_iter().filter(|dir| dir.is_dir()).collect()
}

fn emit_cuda_link_directives() {
    let cuda_lib_dirs = discover_cuda_lib_dirs();
    if cuda_lib_dirs.is_empty() {
        panic!(
            "qwen3-tts.cpp CUDA build requested but no CUDA library directories were found; set CUDA_HOME/CUDA_PATH or install the CUDA toolkit"
        );
    }

    for cuda_lib_dir in &cuda_lib_dirs {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }

    let required_cuda_libs = ["cudart", "cublas", "cuda"];
    for library in required_cuda_libs {
        if !cuda_lib_dirs
            .iter()
            .any(|dir| contains_cuda_library(dir, library))
        {
            panic!(
                "qwen3-tts.cpp CUDA build requested but `{library}` was not found in any CUDA library directory: {}",
                cuda_lib_dirs
                    .iter()
                    .map(|dir| dir.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}

fn contains_cuda_library(dir: &Path, stem: &str) -> bool {
    let prefixes = if cfg!(target_os = "windows") {
        vec![format!("{stem}.lib"), format!("{stem}.dll")]
    } else if cfg!(target_os = "macos") {
        vec![format!("lib{stem}.dylib")]
    } else {
        vec![format!("lib{stem}.so"), format!("lib{stem}.so.")]
    };

    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };

    entries.flatten().any(|entry| {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        prefixes.iter().any(|prefix| file_name.starts_with(prefix))
    })
}
