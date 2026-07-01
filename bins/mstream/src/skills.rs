use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use include_dir::{include_dir, Dir};
use motlie_vfs::core::{FsAccess, FsObserver, FsServer};

pub const SKILLS_DIR: &str = env!("MSTREAM_SKILLS_DIR");
pub static SKILLS: Dir<'static> = include_dir!("$MSTREAM_SKILLS_DIR");

#[cfg(target_os = "linux")]
pub struct SkillsMount {
    mount: motlie_vfs::client::local::LocalMount,
    backing_dir: PathBuf,
    mountpoint: PathBuf,
    remove_mountpoint: bool,
}

#[cfg(target_os = "linux")]
impl SkillsMount {
    fn unmount(self) -> Result<()> {
        let backing_dir = self.backing_dir.clone();
        let mountpoint = self.mountpoint.clone();
        let remove_mountpoint = self.remove_mountpoint;
        let result = self.mount.unmount();
        clean_owned_backing_dir(&backing_dir);
        if remove_mountpoint {
            clean_created_mountpoint(&mountpoint);
        }
        result
    }
}

#[cfg(not(target_os = "linux"))]
pub struct SkillsMount;

pub fn mount_skills(mountpoint: Option<PathBuf>) -> Option<SkillsMount> {
    let Some(mountpoint) = mountpoint else {
        eprintln!(
            "skills FUSE unavailable: no --mount-skill specified; continuing without mounting skills"
        );
        return None;
    };
    match try_mount_skills(mountpoint) {
        Ok(Some(mount)) => Some(mount),
        Ok(None) => None,
        Err(err) => {
            eprintln!("skills FUSE unavailable: {err}; continuing without mount");
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn try_mount_skills(mountpoint: PathBuf) -> Result<Option<SkillsMount>> {
    let mountpoint = normalize_mountpoint(mountpoint)?;
    let remove_mountpoint = !mountpoint.exists();
    let backing_dir = create_backing_dir_for_mountpoint(&mountpoint)
        .context("failed to create embedded skills backing directory")?;

    let owner = current_owner();
    let server = FsServer::builder()
        .mount("skills", backing_dir.clone(), true)
        .overlay(true)
        .observer(SkillAccessObserver)
        .build()
        .context("failed to build embedded skills VFS server")?;
    server
        .overlay()
        .context("embedded skills VFS server was built without overlay support")?
        .put_static_layer("project-skills", 50, "skills", &SKILLS, owner)
        .with_context(|| format!("failed to register embedded skills from {SKILLS_DIR}"))?;

    let mount =
        match motlie_vfs::client::local::mount_local(Arc::new(server), "skills", &mountpoint)
            .with_context(|| {
                format!(
                    "failed to mount embedded skills at {}",
                    mountpoint.display()
                )
            }) {
            Ok(mount) => mount,
            Err(err) => {
                clean_owned_backing_dir(&backing_dir);
                if remove_mountpoint {
                    clean_created_mountpoint(&mountpoint);
                }
                return Err(err);
            }
        };
    eprintln!("skills FUSE mounted at {}", mount.mountpoint().display());
    Ok(Some(SkillsMount {
        mount,
        backing_dir,
        mountpoint,
        remove_mountpoint,
    }))
}

#[cfg(not(target_os = "linux"))]
fn try_mount_skills(_mountpoint: PathBuf) -> Result<Option<SkillsMount>> {
    eprintln!("skills FUSE unavailable on this host platform; continuing without mount");
    Ok(None)
}

#[cfg(target_os = "linux")]
pub fn unmount_skills(mount: Option<SkillsMount>) {
    if let Some(mount) = mount {
        if let Err(err) = mount.unmount() {
            eprintln!("failed to unmount skills FUSE: {err}");
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn unmount_skills(_mount: Option<SkillsMount>) {}

#[cfg(target_os = "linux")]
fn normalize_mountpoint(mountpoint: PathBuf) -> Result<PathBuf> {
    let absolute = if mountpoint.is_absolute() {
        mountpoint
    } else {
        std::env::current_dir()
            .context("failed to resolve daemon working directory")?
            .join(mountpoint)
    };
    let absolute = canonicalize_mountpoint(&absolute)?;
    let cwd = std::env::current_dir()
        .context("failed to resolve daemon working directory")?
        .canonicalize()
        .context("failed to canonicalize daemon working directory")?;
    if cwd.starts_with(&absolute) {
        anyhow::bail!(
            "refusing to mount embedded skills at {} because it is the daemon working directory or one of its parents",
            absolute.display()
        );
    }
    Ok(absolute)
}

#[cfg(target_os = "linux")]
fn canonicalize_mountpoint(mountpoint: &Path) -> Result<PathBuf> {
    if mountpoint.exists() {
        return mountpoint
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", mountpoint.display()));
    }
    let Some(parent) = mountpoint.parent() else {
        return Ok(mountpoint.to_path_buf());
    };
    if parent.as_os_str().is_empty() || !parent.exists() {
        return Ok(mountpoint.to_path_buf());
    }
    let parent = parent
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", parent.display()))?;
    Ok(match mountpoint.file_name() {
        Some(name) => parent.join(name),
        None => parent,
    })
}

#[cfg(target_os = "linux")]
fn create_backing_dir_for_mountpoint(mountpoint: &Path) -> std::io::Result<PathBuf> {
    let parent = mountpoint
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let pid = std::process::id();
    for attempt in 0..1000 {
        let candidate = parent.join(format!(".mstream-skills-backing-{pid}-{attempt}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!(
            "could not allocate backing directory under {}",
            parent.display()
        ),
    ))
}

#[cfg(target_os = "linux")]
fn clean_owned_backing_dir(backing_dir: &Path) {
    match std::fs::remove_dir_all(backing_dir) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => eprintln!(
            "failed to remove skills backing dir {}: {err}",
            backing_dir.display()
        ),
    }
}

#[cfg(target_os = "linux")]
fn clean_created_mountpoint(mountpoint: &Path) {
    match std::fs::remove_dir(mountpoint) {
        Ok(()) => {}
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
            ) => {}
        Err(err) => eprintln!(
            "failed to remove empty skills mountpoint {}: {err}",
            mountpoint.display()
        ),
    }
}

#[cfg(target_os = "linux")]
fn current_owner() -> (u32, u32) {
    unsafe { (libc::geteuid(), libc::getegid()) }
}

struct SkillAccessObserver;

impl FsObserver for SkillAccessObserver {
    fn on_access(&self, access: &FsAccess<'_>) {
        tracing::info!(
            target: "mstream.skills_vfs",
            op = ?access.op,
            tag = access.tag,
            path = access.path,
            bytes = access.bytes,
            errno = access.errno,
            latency_ms = access.latency.as_millis(),
            "skills VFS access"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_are_embedded() {
        assert!(SKILLS.get_file("SKILL.md").is_some());
    }

    #[test]
    fn skills_dir_is_supplied_by_build_script() {
        let path = Path::new(SKILLS_DIR);
        assert!(path.is_absolute());
        assert!(path.ends_with(".agents/skills/project"));
    }

    #[test]
    fn omitted_mountpoint_does_not_mount_skills() {
        assert!(mount_skills(None).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mountpoint_covering_daemon_cwd_is_rejected() {
        let cwd = std::env::current_dir().expect("cwd");
        let err = normalize_mountpoint(cwd).expect_err("cwd mountpoint rejected");
        assert!(err
            .to_string()
            .contains("refusing to mount embedded skills"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn relative_mountpoint_is_normalized_to_absolute_path() {
        let path = normalize_mountpoint(PathBuf::from("__mstream_mount_test__"))
            .expect("relative mountpoint");
        assert!(path.is_absolute());
        assert!(path.ends_with("__mstream_mount_test__"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backing_dir_is_unique_sibling_of_mountpoint() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mountpoint = tempdir.path().join("skills-backing");
        let backing = create_backing_dir_for_mountpoint(&mountpoint).expect("backing dir");

        assert_ne!(backing, mountpoint);
        assert_eq!(backing.parent(), Some(tempdir.path()));
        assert!(backing
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".mstream-skills-backing-")));
        assert!(backing.exists());

        clean_owned_backing_dir(&backing);
        assert!(!backing.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cleanup_removes_created_mountpoint_only_when_empty() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mountpoint = tempdir.path().join("skills");
        std::fs::create_dir(&mountpoint).expect("mountpoint");
        clean_created_mountpoint(&mountpoint);
        assert!(!mountpoint.exists());

        let non_empty = tempdir.path().join("non-empty");
        std::fs::create_dir(&non_empty).expect("non-empty mountpoint");
        std::fs::write(non_empty.join("keep"), b"data").expect("write marker");
        clean_created_mountpoint(&non_empty);
        assert!(non_empty.exists());
    }
}
