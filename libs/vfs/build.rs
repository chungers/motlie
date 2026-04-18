use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");

    let git_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let build_time_utc = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|ts| !ts.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=MOTLIE_VFS_BUILD_GIT_SHA={git_sha}");
    println!("cargo:rustc-env=MOTLIE_VFS_BUILD_TIME_UTC={build_time_utc}");
}
