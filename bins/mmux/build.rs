use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

fn main() {
    for path in git_rerun_paths() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=MMUX_GIT_SHA");
    println!("cargo:rerun-if-env-changed=MMUX_BUILD_DATE");

    let sha = std::env::var("MMUX_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(current_git_sha);
    let build_date = std::env::var("MMUX_BUILD_DATE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(current_build_date);

    println!("cargo:rustc-env=MMUX_GIT_SHA={sha}");
    println!("cargo:rustc-env=MMUX_BUILD_DATE={build_date}");
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
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_else(|| ".".into()));
    let repo_git = manifest_dir.join("../../.git");
    let git_dir = resolve_git_dir(&repo_git);
    let Ok(head_content) = fs::read_to_string(git_dir.join("HEAD")) else {
        return "unknown".to_string();
    };
    let head = head_content.trim();

    if let Some(ref_path) = head.strip_prefix("ref: ") {
        read_git_ref(&git_dir, ref_path).unwrap_or_else(|| "unknown".to_string())
    } else if is_git_sha(head) {
        head.to_string()
    } else {
        "unknown".to_string()
    }
}

fn read_git_ref(git_dir: &Path, ref_path: &str) -> Option<String> {
    for base in git_ref_bases(git_dir) {
        if let Some(value) = read_loose_ref(&base.join(ref_path)) {
            return Some(value);
        }
        if let Some(value) = read_packed_ref(&base.join("packed-refs"), ref_path) {
            return Some(value);
        }
    }
    None
}

fn git_ref_bases(git_dir: &Path) -> Vec<PathBuf> {
    let mut bases = vec![git_dir.to_path_buf()];
    if let Some(common_dir) = common_git_dir(git_dir) {
        bases.push(common_dir);
    }
    bases.sort();
    bases.dedup();
    bases
}

fn read_loose_ref(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| is_git_sha(value))
}

fn read_packed_ref(path: &Path, ref_path: &str) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    content.lines().find_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            return None;
        }
        let mut parts = line.split_whitespace();
        let sha = parts.next()?;
        let name = parts.next()?;
        (name == ref_path && is_git_sha(sha)).then(|| sha.to_string())
    })
}

fn is_git_sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn current_build_date() -> String {
    let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "unknown".to_string();
    };
    let days = (duration.as_secs() / 86_400) as i64;
    let (year, month, day) = civil_from_unix_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

fn civil_from_unix_days(days: i64) -> (i64, u32, u32) {
    let shifted_days = days + 719_468;
    let era = shifted_days.div_euclid(146_097);
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = month_index + if month_index < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}
