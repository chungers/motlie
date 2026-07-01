use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use include_dir::{include_dir, Dir};
use motlie_vfs::core::{FsAccess, FsObserver, FsServer};

pub static PROJECT_SKILLS: Dir<'static> =
    include_dir!("$CARGO_MANIFEST_DIR/../../.agents/skills/project");

#[cfg(target_os = "linux")]
pub type SkillsMount = motlie_vfs::client::local::LocalMount;

#[cfg(not(target_os = "linux"))]
pub struct SkillsMount;

pub fn mount_project_skills() -> Option<SkillsMount> {
    match try_mount_project_skills() {
        Ok(Some(mount)) => Some(mount),
        Ok(None) => None,
        Err(err) => {
            eprintln!("skills FUSE unavailable: {err}; continuing without mount");
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn try_mount_project_skills() -> Result<Option<SkillsMount>> {
    let Some(mountpoint) = default_mountpoint() else {
        eprintln!("skills FUSE unavailable: XDG_RUNTIME_DIR is not set; continuing without mount");
        return Ok(None);
    };
    let backing_dir = backing_dir_for_mountpoint(&mountpoint)
        .context("failed to choose embedded skills backing directory")?;
    std::fs::create_dir_all(&backing_dir)
        .with_context(|| format!("failed to create {}", backing_dir.display()))?;

    let owner = current_owner();
    let server = FsServer::builder()
        .mount("skills", backing_dir, false)
        .overlay(true)
        .observer(SkillAccessObserver)
        .build()
        .context("failed to build embedded skills VFS server")?;
    server
        .overlay()
        .context("embedded skills VFS server was built without overlay support")?
        .put_static_layer("project-skills", 50, "skills", &PROJECT_SKILLS, owner)
        .context("failed to register embedded project skills")?;

    let mount = motlie_vfs::client::local::mount_local(Arc::new(server), "skills", &mountpoint)
        .with_context(|| {
            format!(
                "failed to mount embedded skills at {}",
                mountpoint.display()
            )
        })?;
    eprintln!("skills FUSE mounted at {}", mount.mountpoint().display());
    Ok(Some(mount))
}

#[cfg(not(target_os = "linux"))]
fn try_mount_project_skills() -> Result<Option<SkillsMount>> {
    eprintln!("skills FUSE unavailable on this host platform; continuing without mount");
    Ok(None)
}

#[cfg(target_os = "linux")]
pub fn unmount_project_skills(mount: Option<SkillsMount>) {
    if let Some(mount) = mount {
        if let Err(err) = mount.unmount() {
            eprintln!("failed to unmount skills FUSE: {err}");
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn unmount_project_skills(_mount: Option<SkillsMount>) {}

#[cfg(target_os = "linux")]
fn default_mountpoint() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .map(mountpoint_from_runtime)
}

#[cfg(target_os = "linux")]
fn mountpoint_from_runtime(runtime: PathBuf) -> PathBuf {
    runtime.join("mstream").join("skills")
}

#[cfg(target_os = "linux")]
fn backing_dir_for_mountpoint(mountpoint: &Path) -> Option<PathBuf> {
    mountpoint
        .parent()
        .map(|parent| parent.join("skills-backing"))
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
    fn project_skills_are_embedded() {
        assert!(PROJECT_SKILLS.get_file("SKILL.md").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mountpoint_uses_runtime_dir() {
        assert_eq!(
            mountpoint_from_runtime(PathBuf::from("/tmp/mstream-test-runtime")),
            PathBuf::from("/tmp/mstream-test-runtime/mstream/skills")
        );
    }
}
