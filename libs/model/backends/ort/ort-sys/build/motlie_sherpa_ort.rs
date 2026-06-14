use std::{
    env,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

use bzip2::read::BzDecoder;
use tar::Archive;

use crate::{
    download,
    error::{Error, ResultExt},
    internal, log,
};

const RELEASE_BASE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download";
const SHERPA_ONNX_VERSION: &str = "1.13.2";
const ONNXRUNTIME_VERSION: &str = "1.24.2";
const ONNXRUNTIME_TAG: &str = "v1.24.2";
const ONNXRUNTIME_STATIC_CUDA_PATCH: &str = "build/patches/onnxruntime-v1.24.2-static-cuda.patch";
const SOURCE_SHERPA_ONNX: &str = "sherpa-onnx";
const SOURCE_PYKE: &str = "pyke";
const MOTLIE_ORT_CUDA_STATIC_LIB_DIR: &str = "MOTLIE_ORT_CUDA_STATIC_LIB_DIR";
const MOTLIE_ORT_CUDA_ARCH: &str = "MOTLIE_ORT_CUDA_ARCH";
const ORT_STATIC_CUDA_TARGETS: &[&str] = &[
    "onnxruntime_common",
    "onnxruntime_flatbuffers",
    "onnxruntime_framework",
    "onnxruntime_graph",
    "onnxruntime_lora",
    "onnxruntime_mlas",
    "onnxruntime_optimizer",
    "onnxruntime_providers",
    "onnxruntime_session",
    "onnxruntime_util",
    "onnxruntime_providers_cuda",
    "cpuinfo",
    "re2",
];

pub fn resolve_lib_dir() -> Result<Option<PathBuf>, Error> {
    let source = env::var("MOTLIE_ORT_SOURCE").unwrap_or_else(|_| SOURCE_SHERPA_ONNX.to_string());
    match source.as_str() {
        SOURCE_PYKE => return Ok(None),
        SOURCE_SHERPA_ONNX => {}
        other => {
            return Err(Error::new(format!(
                "unsupported MOTLIE_ORT_SOURCE `{other}`; expected `{SOURCE_SHERPA_ONNX}` or `{SOURCE_PYKE}`"
            )));
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    if cuda_feature_enabled() {
        return resolve_cuda_static_lib_dir(&target_os, &target_arch).map(Some);
    }

    let archive_name = match archive_name(&target_os, &target_arch) {
        Some(archive_name) => archive_name,
        None => {
            log::warning!(
                "Motlie sherpa-onnx ORT source is not mapped for target os `{}` arch `{}`; falling back to Pyke ORT",
                target_os,
                target_arch
            );
            return Ok(None);
        }
    };
    let archive_stem = archive_name.trim_end_matches(".tar.bz2");

    let cache_root = internal::dirs::cache_dir()
        .ok_or_else(|| Error::new("could not determine cache directory"))?
        .join("motlie-sherpa-ort")
        .join(format!("v{SHERPA_ONNX_VERSION}"))
        .join(env::var("TARGET")?);
    let extracted_dir = cache_root.join(archive_stem);
    let lib_dir = extracted_dir.join("lib");
    if lib_dir.join(static_library_name("onnxruntime")).is_file() {
        log::debug!(
            "Using Motlie sherpa-onnx ONNX Runtime from {}",
            lib_dir.display()
        );
        return Ok(Some(lib_dir));
    }

    fs::create_dir_all(&cache_root)?;
    let archive_path = cache_root.join(&archive_name);
    if !archive_path.is_file() {
        if let Some(local_archive_dir) = env::var_os("SHERPA_ONNX_ARCHIVE_DIR") {
            let local_archive_path = PathBuf::from(local_archive_dir).join(&archive_name);
            copy_file_atomically(&local_archive_path, &archive_path)?;
        } else {
            let url = format!("{RELEASE_BASE_URL}/v{SHERPA_ONNX_VERSION}/{archive_name}");
            log::debug!("Downloading Motlie sherpa-onnx ONNX Runtime source from {url}");
            let mut reader = download::fetch_file(&url)?;
            write_reader_atomically(&mut reader, &archive_path)?;
        }
    }

    if extracted_dir.exists() {
        fs::remove_dir_all(&extracted_dir)?;
    }

    let tar_file = File::open(&archive_path)?;
    let decoder = BzDecoder::new(tar_file);
    let mut archive = Archive::new(decoder);
    archive.unpack(&cache_root)?;

    if !lib_dir.join(static_library_name("onnxruntime")).is_file() {
        return Err(Error::new(format!(
            "sherpa-onnx archive did not contain expected ONNX Runtime archive in `{}`",
            lib_dir.display()
        )));
    }

    log::debug!(
        "Using Motlie sherpa-onnx ONNX Runtime from {}",
        lib_dir.display()
    );
    Ok(Some(lib_dir))
}

fn cuda_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_CUDA").is_some()
}

fn resolve_cuda_static_lib_dir(target_os: &str, target_arch: &str) -> Result<PathBuf, Error> {
    if target_os != "linux" {
        return Err(Error::new(format!(
            "MOTLIE_ORT_SOURCE={SOURCE_SHERPA_ONNX} with ort/cuda is only mapped for Linux targets; target was {target_os}/{target_arch}"
        )));
    }

    if let Some(lib_dir) = env::var_os(MOTLIE_ORT_CUDA_STATIC_LIB_DIR) {
        let lib_dir = PathBuf::from(lib_dir);
        validate_cuda_static_lib_dir(&lib_dir)?;
        log::debug!(
            "Using Motlie source-built static CUDA ONNX Runtime from {}",
            lib_dir.display()
        );
        return Ok(lib_dir);
    }

    let cuda_home = find_cuda_home(target_arch).ok_or_else(|| {
        Error::new(format!(
            "MOTLIE_ORT_SOURCE={SOURCE_SHERPA_ONNX} with ort/cuda requires a CUDA toolkit; set CUDA_HOME or CUDA_PATH if it is not under /usr/local/cuda"
        ))
    })?;
    let cudnn9 = find_cudnn9(&cuda_home, target_arch).ok_or_else(|| {
        Error::new(format!(
            "MOTLIE_ORT_SOURCE={SOURCE_SHERPA_ONNX} with ort/cuda requires cuDNN 9 headers and library before Motlie can build/link static CUDA ONNX Runtime; CUDA detected at `{}`; cuDNN 9 was not found. On DGX/SBSA hosts the Ubuntu multiarch paths /usr/include/aarch64-linux-gnu and /usr/lib/aarch64-linux-gnu are checked; set CUDNN_HOME/CUDNN_PATH for a custom install.",
            cuda_home.display()
        ))
    })?;

    build_cuda_static_ort(target_arch, &cuda_home, &cudnn9)
}

fn build_cuda_static_ort(
    target_arch: &str,
    cuda_home: &Path,
    cudnn9: &(PathBuf, PathBuf),
) -> Result<PathBuf, Error> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let ort_source_dir = manifest_dir
        .parent()
        .ok_or_else(|| Error::new(format!("{} has no parent", manifest_dir.display())))?
        .join("vendor")
        .join("onnxruntime");
    validate_onnxruntime_source_dir(&ort_source_dir)?;

    let patch_path = manifest_dir.join(ONNXRUNTIME_STATIC_CUDA_PATCH);
    let patch_bytes = fs::read(&patch_path)
        .with_context(|| format!("failed to read {}", patch_path.display()))?;
    let patch_checksum = stable_checksum(&patch_bytes);

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let source_dir = out_dir.join("onnxruntime-v1.24.2-static-cuda-src");
    let build_dir = out_dir.join("onnxruntime-v1.24.2-static-cuda-build");
    let lib_dir = build_dir.join("Release");
    let stamp_path = build_dir.join("motlie-static-cuda.stamp");
    let cudnn_library = cudnn_library_path(&cudnn9.1)?;
    let cuda_arch = cuda_architecture();
    let stamp = format!(
        "onnxruntime_tag={ONNXRUNTIME_TAG}\ntarget_arch={target_arch}\ncuda_home={}\ncudnn_include={}\ncudnn_library={}\ncuda_arch={cuda_arch}\npatch_checksum={patch_checksum:016x}\n",
        cuda_home.display(),
        cudnn9.0.display(),
        cudnn_library.display()
    );

    let stamp_matches =
        matches!(fs::read_to_string(&stamp_path), Ok(existing) if existing == stamp);
    if stamp_matches && validate_cuda_static_lib_dir(&lib_dir).is_ok() {
        log::debug!(
            "Using cached Motlie source-built static CUDA ONNX Runtime from {}",
            lib_dir.display()
        );
        return Ok(lib_dir);
    }

    if stamp_path.is_file() && !stamp_matches {
        if build_dir.exists() {
            fs::remove_dir_all(&build_dir)
                .with_context(|| format!("failed to remove {}", build_dir.display()))?;
        }
        if source_dir.exists() {
            fs::remove_dir_all(&source_dir)
                .with_context(|| format!("failed to remove {}", source_dir.display()))?;
        }
    }

    fs::create_dir_all(&out_dir)?;
    if !source_dir.exists() {
        clone_onnxruntime_source(&ort_source_dir, &source_dir)?;
        apply_onnxruntime_static_cuda_patch(&source_dir, &patch_path)?;
    }
    run_onnxruntime_static_cuda_build(
        &source_dir,
        &build_dir,
        target_arch,
        cuda_home,
        &cudnn9.0,
        &cudnn_library,
        &cuda_arch,
    )?;
    create_cuda_device_link_archive(&lib_dir, target_arch, cuda_home, &cuda_arch)?;
    validate_cuda_static_lib_dir(&lib_dir)?;
    fs::create_dir_all(&build_dir)?;
    fs::write(&stamp_path, stamp)
        .with_context(|| format!("failed to write {}", stamp_path.display()))?;

    log::debug!(
        "Built Motlie source-built static CUDA ONNX Runtime from {} into {}",
        ort_source_dir.display(),
        lib_dir.display()
    );
    Ok(lib_dir)
}

fn validate_onnxruntime_source_dir(source_dir: &Path) -> Result<(), Error> {
    let build_script = source_dir.join("tools").join("ci_build").join("build.py");
    if !build_script.is_file() {
        return Err(Error::new(format!(
            "ONNX Runtime submodule is missing {}; initialize libs/model/backends/ort/vendor/onnxruntime at {ONNXRUNTIME_TAG}",
            build_script.display()
        )));
    }

    let version_file = source_dir.join("VERSION_NUMBER");
    let version = fs::read_to_string(&version_file)
        .with_context(|| format!("failed to read {}", version_file.display()))?;
    if version.trim() != ONNXRUNTIME_VERSION {
        return Err(Error::new(format!(
            "ONNX Runtime submodule version mismatch: expected {ONNXRUNTIME_VERSION} ({ONNXRUNTIME_TAG}), found `{}` in {}",
            version.trim(),
            version_file.display()
        )));
    }

    Ok(())
}

fn clone_onnxruntime_source(source_dir: &Path, build_source_dir: &Path) -> Result<(), Error> {
    let parent = build_source_dir
        .parent()
        .ok_or_else(|| Error::new(format!("{} has no parent", build_source_dir.display())))?;
    fs::create_dir_all(parent)?;

    run_command(
        Command::new("git")
            .arg("clone")
            .arg("--shared")
            .arg("--no-checkout")
            .arg(source_dir)
            .arg(build_source_dir),
        "clone ONNX Runtime submodule for static CUDA build",
    )?;
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(build_source_dir)
            .arg("checkout")
            .arg("--detach")
            .arg(ONNXRUNTIME_TAG),
        "checkout ONNX Runtime v1.24.2 for static CUDA build",
    )
}

fn apply_onnxruntime_static_cuda_patch(source_dir: &Path, patch_path: &Path) -> Result<(), Error> {
    run_command(
        Command::new("git")
            .arg("-C")
            .arg(source_dir)
            .arg("apply")
            .arg("--whitespace=nowarn")
            .arg(patch_path),
        "apply Motlie ONNX Runtime static CUDA patch",
    )
}

fn run_onnxruntime_static_cuda_build(
    source_dir: &Path,
    build_dir: &Path,
    target_arch: &str,
    cuda_home: &Path,
    cudnn_include_dir: &Path,
    cudnn_library: &Path,
    cuda_arch: &str,
) -> Result<(), Error> {
    let build_script = source_dir.join("tools").join("ci_build").join("build.py");
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "8".to_owned());
    let cudnn_home = cudnn_home_for_build(cudnn_include_dir, cudnn_library);
    let cuda_lib_dir = cuda_lib_dir(cuda_home, target_arch);

    let mut command = Command::new("python3");
    command
        .arg(build_script)
        .arg("--build_dir")
        .arg(build_dir)
        .arg("--config")
        .arg("Release")
        .arg("--update")
        .arg("--build")
        .arg("--parallel")
        .arg(jobs)
        .arg("--skip_tests")
        .arg("--compile_no_warning_as_error")
        .arg("--use_cuda")
        .arg("--cuda_home")
        .arg(cuda_home)
        .arg("--cudnn_home")
        .arg(&cudnn_home);
    for target in ORT_STATIC_CUDA_TARGETS {
        command.arg("--target").arg(target);
    }
    command
        .arg("--cmake_extra_defines")
        .arg(format!("CMAKE_CUDA_ARCHITECTURES={cuda_arch}"))
        .arg("onnxruntime_BUILD_CUDA_EP_STATIC_LIB=ON")
        .arg("onnxruntime_BUILD_SHARED_LIB=OFF")
        .arg("onnxruntime_BUILD_UNIT_TESTS=OFF")
        .arg(format!("CUDNN_INCLUDE_DIR={}", cudnn_include_dir.display()))
        .arg(format!("cudnn_LIBRARY={}", cudnn_library.display()))
        .arg("onnxruntime_USE_CUDA_NHWC_OPS=OFF")
        .env("CUDA_HOME", cuda_home)
        .env("CUDA_PATH", cuda_home)
        .env("CUDNN_HOME", &cudnn_home)
        .env("CUDNN_PATH", &cudnn_home)
        .env("CUDNN_INCLUDE_DIR", cudnn_include_dir)
        .env("cudnn_LIBRARY", cudnn_library)
        .env("LIBRARY_PATH", cuda_lib_dir);

    run_command(&mut command, "build ONNX Runtime static CUDA targets")
}

fn create_cuda_device_link_archive(
    lib_dir: &Path,
    target_arch: &str,
    cuda_home: &Path,
    cuda_arch: &str,
) -> Result<(), Error> {
    fs::create_dir_all(lib_dir)?;
    let cuda_lib_dir = cuda_lib_dir(cuda_home, target_arch);
    let object_path = lib_dir.join("motlie_ort_cuda_dlink.o");
    let archive_path = lib_dir.join("libmotlie_ort_cuda_dlink.a");

    for lib in ["libcufft_static.a", "libcudart_static.a", "libcudadevrt.a"] {
        let path = cuda_lib_dir.join(lib);
        if !path.is_file() {
            return Err(Error::new(format!(
                "CUDA static device-link prerequisite is missing: {}",
                path.display()
            )));
        }
    }

    run_command(
        Command::new(cuda_home.join("bin").join("nvcc"))
            .arg("-dlink")
            .arg(format!("-arch=sm_{cuda_arch}"))
            .arg("-o")
            .arg(&object_path)
            .arg(cuda_lib_dir.join("libcufft_static.a"))
            .arg(cuda_lib_dir.join("libcudart_static.a"))
            .arg(cuda_lib_dir.join("libcudadevrt.a")),
        "create CUDA device-link object for static cuFFT",
    )?;

    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    run_command(
        Command::new(ar)
            .arg("rcs")
            .arg(&archive_path)
            .arg(&object_path),
        "archive CUDA device-link object",
    )
}

fn cudnn_library_path(lib_dir: &Path) -> Result<PathBuf, Error> {
    for name in ["libcudnn.so.9", "libcudnn.so"] {
        let candidate = lib_dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Error::new(format!(
        "cuDNN 9 library directory did not contain libcudnn.so.9 or libcudnn.so: {}",
        lib_dir.display()
    )))
}

fn cudnn_home_for_build(include_dir: &Path, library: &Path) -> PathBuf {
    for var in ["CUDNN_HOME", "CUDNN_PATH"] {
        if let Some(path) = env::var_os(var) {
            return PathBuf::from(path);
        }
    }

    if include_dir.starts_with("/usr/include") && library.starts_with("/usr/lib") {
        return PathBuf::from("/usr");
    }

    include_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| include_dir.to_path_buf())
}

fn cuda_architecture() -> String {
    env::var(MOTLIE_ORT_CUDA_ARCH).unwrap_or_else(|_| "121".to_owned())
}

fn stable_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn run_command(command: &mut Command, action: &str) -> Result<(), Error> {
    log::debug!("{}: {:?}", action, command);
    let status = command
        .status()
        .with_context(|| format!("failed to {action}"))?;
    if status.success() {
        return Ok(());
    }

    Err(Error::new(format!(
        "failed to {action}; command exited with {status}"
    )))
}

fn validate_cuda_static_lib_dir(lib_dir: &Path) -> Result<(), Error> {
    if !lib_dir.is_dir() {
        return Err(Error::new(format!(
            "{MOTLIE_ORT_CUDA_STATIC_LIB_DIR} does not name a directory: {}",
            lib_dir.display()
        )));
    }

    let has_single_ort = lib_dir.join(static_library_name("onnxruntime")).is_file();
    let has_split_ort = lib_dir
        .join(static_library_name("onnxruntime_common"))
        .is_file();
    if !has_single_ort && !has_split_ort {
        return Err(Error::new(format!(
            "{MOTLIE_ORT_CUDA_STATIC_LIB_DIR} is missing libonnxruntime.a or split ONNX Runtime static archives: {}",
            lib_dir.display()
        )));
    }

    if !lib_dir
        .join(static_library_name("onnxruntime_providers_cuda"))
        .is_file()
    {
        return Err(Error::new(format!(
            "{MOTLIE_ORT_CUDA_STATIC_LIB_DIR} is missing the static CUDA execution provider archive libonnxruntime_providers_cuda.a: {}",
            lib_dir.display()
        )));
    }

    Ok(())
}

fn find_cuda_home(target_arch: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(path) = env::var_os(var) {
            candidates.push(PathBuf::from(path));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda-13.0"));

    candidates.into_iter().find(|candidate| {
        candidate.join("bin").join("nvcc").is_file()
            || cuda_lib_dir(candidate, target_arch).is_dir()
    })
}

fn find_cudnn9(cuda_home: &Path, target_arch: &str) -> Option<(PathBuf, PathBuf)> {
    let include_dir = cudnn_include_dirs(cuda_home)
        .into_iter()
        .find(|dir| cudnn_major_is_9(&dir.join("cudnn_version.h")))?;
    let lib_dir = cudnn_lib_dirs(cuda_home, target_arch)
        .into_iter()
        .find(|dir| dir.join("libcudnn.so.9").exists() || dir.join("libcudnn.so").exists())?;
    Some((include_dir, lib_dir))
}

fn cudnn_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["CUDNN_HOME", "CUDNN_PATH"] {
        if let Some(path) = env::var_os(var) {
            roots.push(PathBuf::from(path));
        }
    }
    roots
}

fn cudnn_include_dirs(cuda_home: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for root in cudnn_roots() {
        dirs.push(root.join("include"));
        dirs.push(root);
    }
    dirs.push(cuda_home.join("include"));
    dirs.push(PathBuf::from("/usr/include/aarch64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/include"));
    dirs
}

fn cudnn_lib_dirs(cuda_home: &Path, target_arch: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for root in cudnn_roots() {
        dirs.push(root.join("lib64"));
        dirs.push(root.join("lib"));
        dirs.push(root);
    }
    dirs.push(cuda_lib_dir(cuda_home, target_arch));
    if target_arch == "aarch64" {
        dirs.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    } else if target_arch == "x86_64" {
        dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    }
    dirs
}

fn cuda_lib_dir(cuda_home: &Path, target_arch: &str) -> PathBuf {
    match target_arch {
        "aarch64" => cuda_home.join("targets").join("sbsa-linux").join("lib"),
        "x86_64" => cuda_home.join("targets").join("x86_64-linux").join("lib"),
        _ => cuda_home.join("lib64"),
    }
}

fn cudnn_major_is_9(header: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(header) else {
        return false;
    };
    contents.lines().any(|line| {
        let mut parts = line.split_whitespace();
        matches!(
            (parts.next(), parts.next(), parts.next()),
            (Some("#define"), Some("CUDNN_MAJOR"), Some("9"))
        )
    })
}

fn archive_name(target_os: &str, target_arch: &str) -> Option<String> {
    let version = SHERPA_ONNX_VERSION;
    let name = match (target_os, target_arch) {
        ("linux", "x86_64") => format!("sherpa-onnx-v{version}-linux-x64-static-lib.tar.bz2"),
        ("linux", "aarch64") => {
            format!("sherpa-onnx-v{version}-linux-aarch64-static-lib.tar.bz2")
        }
        ("macos", "x86_64") => format!("sherpa-onnx-v{version}-osx-x64-static-lib.tar.bz2"),
        ("macos", "aarch64") => format!("sherpa-onnx-v{version}-osx-arm64-static-lib.tar.bz2"),
        ("windows", "x86_64") => {
            format!("sherpa-onnx-v{version}-win-x64-static-MT-Release-lib.tar.bz2")
        }
        _ => return None,
    };
    Some(name)
}

fn static_library_name(lib: &str) -> String {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        format!("{lib}.lib")
    } else {
        format!("lib{lib}.a")
    }
}

fn copy_file_atomically(source: &Path, dest: &Path) -> Result<(), Error> {
    if !source.is_file() {
        return Err(Error::new(format!(
            "SHERPA_ONNX_ARCHIVE_DIR does not contain expected archive: {}",
            source.display()
        )));
    }
    let mut reader = File::open(source)?;
    write_reader_atomically(&mut reader, dest)
}

fn write_reader_atomically(reader: &mut impl Read, dest: &Path) -> Result<(), Error> {
    let parent = dest
        .parent()
        .ok_or_else(|| Error::new(format!("path has no parent: {}", dest.display())))?;
    fs::create_dir_all(parent)?;
    let temp = dest.with_extension("tmp");
    let write_result = (|| -> io::Result<()> {
        let mut file = File::create(&temp)?;
        io::copy(reader, &mut file)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&temp);
        return Err(Error::new(format!(
            "failed to write {}: {err}",
            dest.display()
        )));
    }
    fs::rename(&temp, dest)?;
    Ok(())
}
