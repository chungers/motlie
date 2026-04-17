use std::env;

fn main() {
    if cfg!(target_family = "unix") && env::var_os("CARGO_FEATURE_MODEL_QWEN3_TTS_CPP").is_some() {
        let lib_dir = env::var_os("DEP_MOTLIE_QWEN3_TTS_CPP_LIBDIR")
            .or_else(|| {
                env::var_os("DEP_MOTLIE_QWEN3_TTS_CPP_ROOT").map(|root| {
                    let mut path = std::path::PathBuf::from(root);
                    path.push("lib");
                    path.into_os_string()
                })
            });
        if let Some(lib_dir) = lib_dir {
            let lib_dir = lib_dir.to_string_lossy();
            let arg = format!("-Wl,-rpath,{lib_dir}");
            println!("cargo:rustc-link-arg-examples={arg}");
            println!("cargo:rustc-link-arg-bins={arg}");
        }
    }
}
