use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ESPEAK_NG_LIB_DIR");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let lib_dir = out_dir.join("lib");
    let library = match find_espeak_library() {
        Some(path) => path,
        None => panic!(
            "failed to locate libespeak-ng. Install libespeak-ng-dev or set ESPEAK_NG_LIB_DIR \
             to a directory containing libespeak-ng.so/libespeak-ng.dylib"
        ),
    };

    if let Some(parent) = library.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }

    if has_linker_name(&library) {
        println!("cargo:rustc-link-lib=dylib=espeak-ng");
        return;
    }

    fs::create_dir_all(&lib_dir).expect("failed to create OUT_DIR/lib for espeak-ng link shim");
    let shim_name = if cfg!(target_os = "macos") {
        "libespeak-ng.dylib"
    } else {
        "libespeak-ng.so"
    };
    let shim_path = lib_dir.join(shim_name);
    if shim_path.exists() {
        fs::remove_file(&shim_path).expect("failed to replace stale espeak-ng link shim");
    }

    create_link_shim(&library, &shim_path);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=espeak-ng");
}

fn find_espeak_library() -> Option<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(explicit) = env::var_os("ESPEAK_NG_LIB_DIR") {
        dirs.push(PathBuf::from(explicit));
    }

    dirs.extend([
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/lib"),
        PathBuf::from("/usr/lib/aarch64-linux-gnu"),
        PathBuf::from("/lib/aarch64-linux-gnu"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
        PathBuf::from("/opt/homebrew/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/lib64"),
    ]);

    dirs.into_iter().find_map(|dir| locate_library_in_dir(&dir))
}

fn locate_library_in_dir(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut fallback = None;

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if name == "libespeak-ng.so" || name == "libespeak-ng.dylib" {
            return Some(path);
        }

        if name.starts_with("libespeak-ng.so.") && fallback.is_none() {
            fallback = Some(path);
        }
    }

    fallback
}

fn has_linker_name(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("libespeak-ng.so" | "libespeak-ng.dylib")
    )
}

fn create_link_shim(target: &Path, shim_path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        if symlink(target, shim_path).is_ok() {
            return;
        }
    }

    fs::copy(target, shim_path).expect("failed to create espeak-ng link shim");
}
