use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=MSTREAM_BUILD_GIT_SHA");
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let sha = std::env::var("MSTREAM_BUILD_GIT_SHA").unwrap_or_else(|_| current_git_sha());
    println!("cargo:rustc-env=MSTREAM_BUILD_GIT_SHA={sha}");
}

fn current_git_sha() -> String {
    let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if is_git_sha(&sha) {
        sha
    } else {
        "unknown".to_string()
    }
}

fn is_git_sha(value: &str) -> bool {
    matches!(value.len(), 7..=40) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
