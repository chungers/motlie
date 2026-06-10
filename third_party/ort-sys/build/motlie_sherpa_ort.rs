use std::{
    env,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use bzip2::read::BzDecoder;
use tar::Archive;

use crate::{download, error::Error, internal, log};

const RELEASE_BASE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download";
const SHERPA_ONNX_VERSION: &str = "1.13.2";
const SOURCE_SHERPA_ONNX: &str = "sherpa-onnx";
const SOURCE_PYKE: &str = "pyke";

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
