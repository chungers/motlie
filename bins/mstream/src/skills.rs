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
}

#[cfg(target_os = "linux")]
impl SkillsMount {
    fn unmount(self) -> Result<()> {
        let backing_dir = self.backing_dir.clone();
        let result = self.mount.unmount();
        clean_empty_backing_dir(&backing_dir);
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
    let backing_dir = backing_dir_for_mountpoint(&mountpoint)
        .context("failed to choose embedded skills backing directory")?;
    std::fs::create_dir_all(&backing_dir)
        .with_context(|| format!("failed to create {}", backing_dir.display()))?;

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
                clean_empty_backing_dir(&backing_dir);
                return Err(err);
            }
        };
    eprintln!("skills FUSE mounted at {}", mount.mountpoint().display());
    Ok(Some(SkillsMount { mount, backing_dir }))
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
fn backing_dir_for_mountpoint(mountpoint: &Path) -> Option<PathBuf> {
    mountpoint
        .parent()
        .map(|parent| parent.join("skills-backing"))
}

#[cfg(target_os = "linux")]
fn clean_empty_backing_dir(backing_dir: &Path) {
    match std::fs::remove_dir(backing_dir) {
        Ok(()) => {}
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
            ) => {}
        Err(err) => eprintln!(
            "failed to remove empty skills backing dir {}: {err}",
            backing_dir.display()
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
    fn backing_dir_uses_explicit_mount_parent() {
        assert_eq!(
            backing_dir_for_mountpoint(Path::new("/tmp/mstream-test/skills")).unwrap(),
            PathBuf::from("/tmp/mstream-test/skills-backing")
        );
    }
}
