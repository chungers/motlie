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

    let mut config = cmake::Config::new(&vendor_dir);
    config.profile("Release");
    config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    config.define("BUILD_SHARED_LIBS", "OFF");

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
    } else {
        config.define("GGML_CUDA", "OFF");
    }

    if cfg!(target_os = "macos") {
        config.define("QWEN3_TTS_COREML", "ON");
        config.define("GGML_METAL", "ON");
    } else {
        config.define("QWEN3_TTS_COREML", "OFF");
    }
    if cfg!(target_env = "gnu") {
        config.define("CMAKE_EXE_LINKER_FLAGS", "-fopenmp");
        config.define("CMAKE_SHARED_LINKER_FLAGS", "-fopenmp");
    }

    let dst = config.build();
    let lib_dir = canonicalize_existing_path(dst.join("lib"));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=qwen3tts");
    println!("cargo:libdir={}", lib_dir.display());
    if cfg!(feature = "cuda") {
        for cuda_lib_dir in discover_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        }
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
}

fn canonicalize_existing_path(path: PathBuf) -> PathBuf {
    fs::canonicalize(&path)
        .unwrap_or_else(|error| panic!("failed to canonicalize `{}`: {error}", path.display()))
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
            dirs.push(root.join("targets/x86_64-linux/lib"));
        }
    }

    dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/sbsa-linux/lib"));
    dirs.push(PathBuf::from("/usr/local/cuda/targets/x86_64-linux/lib"));

    dirs.into_iter().filter(|dir| dir.is_dir()).collect()
}
