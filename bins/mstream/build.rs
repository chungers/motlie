use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=MSTREAM_BUILD_GIT_SHA");
    watch_git_metadata();
    if let Err(err) = configure_skills() {
        panic!("failed to configure embedded skills: {err}");
    }

    let sha = std::env::var("MSTREAM_BUILD_GIT_SHA").unwrap_or_else(|_| current_git_sha());
    println!("cargo:rustc-env=MSTREAM_BUILD_GIT_SHA={sha}");
}

fn watch_git_metadata() {
    let Some(head_path) = git_path("HEAD") else {
        return;
    };
    println!("cargo:rerun-if-changed={}", head_path.display());

    let Some(head_ref) = read_head_ref(&head_path) else {
        return;
    };
    if let Some(ref_path) = git_path(&head_ref) {
        println!("cargo:rerun-if-changed={}", ref_path.display());
    }
    if let Some(packed_refs_path) = git_path("packed-refs") {
        println!("cargo:rerun-if-changed={}", packed_refs_path.display());
    }
}

fn configure_skills() -> io::Result<()> {
    let root = skills_dir()?;
    watch_tree(&root)?;
    println!("cargo:rustc-env=MSTREAM_SKILLS_DIR={}", root.display());
    Ok(())
}

fn skills_dir() -> io::Result<PathBuf> {
    let manifest_dir = match std::env::var_os("CARGO_MANIFEST_DIR") {
        Some(value) => PathBuf::from(value),
        None => {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CARGO_MANIFEST_DIR is not set",
            ))
        }
    };
    Ok(manifest_dir
        .join("../..")
        .canonicalize()?
        .join(".agents/skills/project"))
}

fn watch_tree(root: &Path) -> io::Result<()> {
    let meta = fs::symlink_metadata(root)?;
    if meta.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("embedded skill path is a symlink: {}", root.display()),
        ));
    }
    println!("cargo:rerun-if-changed={}", root.display());
    if !meta.is_dir() {
        return Ok(());
    }

    let mut children = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(|entry| entry.path());
    for child in children {
        watch_tree(&child.path())?;
    }
    Ok(())
}

fn read_head_ref(head_path: &Path) -> Option<String> {
    let head = fs::read_to_string(head_path).ok()?;
    head.strip_prefix("ref:").map(str::trim).and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

fn git_path(path: &str) -> Option<PathBuf> {
    run_git(["rev-parse", "--git-path", path]).map(PathBuf::from)
}

fn current_git_sha() -> String {
    let Some(sha) = run_git(["rev-parse", "HEAD"]) else {
        return "unknown".to_string();
    };
    if is_git_sha(&sha) {
        sha
    } else {
        "unknown".to_string()
    }
}

fn is_git_sha(value: &str) -> bool {
    matches!(value.len(), 7..=40) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn run_git<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
