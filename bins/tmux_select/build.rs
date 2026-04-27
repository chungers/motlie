use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    for path in git_rerun_paths() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=TMUX_SELECT_GIT_SHA");

    let sha = std::env::var("TMUX_SELECT_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(current_git_sha);

    println!("cargo:rustc-env=TMUX_SELECT_GIT_SHA={sha}");
}

fn git_rerun_paths() -> Vec<PathBuf> {
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_else(|| ".".into()));
    let repo_git = manifest_dir.join("../../.git");
    let git_dir = resolve_git_dir(&repo_git);
    let head = git_dir.join("HEAD");
    let mut paths = vec![repo_git, head.clone()];

    if let Ok(head_content) = std::fs::read_to_string(head) {
        if let Some(ref_path) = head_content.trim().strip_prefix("ref: ") {
            paths.push(git_dir.join(ref_path));
            if let Some(common_dir) = common_git_dir(&git_dir) {
                paths.push(common_dir.join(ref_path));
                paths.push(common_dir.join("packed-refs"));
            }
        }
    }

    paths.sort();
    paths.dedup();
    paths
}

fn resolve_git_dir(repo_git: &Path) -> PathBuf {
    if repo_git.is_dir() {
        return repo_git.to_path_buf();
    }

    let Ok(dot_git) = std::fs::read_to_string(repo_git) else {
        return repo_git.to_path_buf();
    };
    let Some(git_dir) = dot_git.trim().strip_prefix("gitdir: ") else {
        return repo_git.to_path_buf();
    };

    let git_dir = PathBuf::from(git_dir);
    if git_dir.is_absolute() {
        git_dir
    } else {
        repo_git
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(git_dir)
    }
}

fn common_git_dir(git_dir: &Path) -> Option<PathBuf> {
    let common_dir = std::fs::read_to_string(git_dir.join("commondir")).ok()?;
    let common_dir = PathBuf::from(common_dir.trim());
    Some(if common_dir.is_absolute() {
        common_dir
    } else {
        git_dir.join(common_dir)
    })
}

fn current_git_sha() -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("rev-parse")
        .arg("HEAD")
        .output();

    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "unknown".to_string(),
    }
}
