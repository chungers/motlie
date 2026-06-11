use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=MSTREAM_BUILD_GIT_SHA");
    watch_git_metadata();

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
