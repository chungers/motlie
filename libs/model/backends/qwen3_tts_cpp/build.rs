use std::env;
use std::path::PathBuf;

fn main() {
    let vendor_dir = PathBuf::from("vendor/qwen3-tts.cpp");

    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/qwen3tts_c_api.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/qwen3tts_c_api.h").display()
    );

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

    let dst = config.build();
    let lib_dir = dst.join("lib");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=qwen3tts");
    println!("cargo:rustc-link-lib=static=qwen3_tts");
    println!("cargo:rustc-link-lib=static=text_tokenizer");
    println!("cargo:rustc-link-lib=static=tts_transformer");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_encoder");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_decoder");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if cfg!(feature = "cuda") {
        for cuda_lib_dir in discover_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        }
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
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
