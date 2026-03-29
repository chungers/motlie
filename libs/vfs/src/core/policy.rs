//! PolicyFn trait and AllowAll default policy.

use super::event::FsOpKind;

/// Policy hook called before every filesystem operation.
///
/// Return `Ok(())` to allow, `Err(errno)` to deny. Called for every `FsOp`
/// including individual `Readdir` entries during policy-filtered directory
/// listing.
pub trait PolicyFn: Send + Sync + 'static {
    fn check(&self, op: FsOpKind, tag: &str, path: &str) -> Result<(), i32>;
}

/// Default policy: allow everything.
pub struct AllowAll;

impl PolicyFn for AllowAll {
    fn check(&self, _op: FsOpKind, _tag: &str, _path: &str) -> Result<(), i32> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_all_permits_everything() {
        let policy = AllowAll;
        assert!(policy.check(FsOpKind::Read, "workspace", "/foo.txt").is_ok());
        assert!(policy.check(FsOpKind::Write, "home", "/.ssh/id_ed25519").is_ok());
        assert!(policy.check(FsOpKind::Unlink, "scratch", "/tmp/junk").is_ok());
    }

    #[test]
    fn custom_deny_policy() {
        struct DenyWrites;
        impl PolicyFn for DenyWrites {
            fn check(&self, op: FsOpKind, _tag: &str, _path: &str) -> Result<(), i32> {
                match op {
                    FsOpKind::Write | FsOpKind::Create | FsOpKind::Mkdir
                    | FsOpKind::Unlink | FsOpKind::Rmdir | FsOpKind::Rename => {
                        Err(libc::EROFS)
                    }
                    _ => Ok(()),
                }
            }
        }

        let policy = DenyWrites;
        assert!(policy.check(FsOpKind::Read, "ws", "/f").is_ok());
        assert!(policy.check(FsOpKind::Lookup, "ws", "/f").is_ok());
        assert_eq!(policy.check(FsOpKind::Write, "ws", "/f"), Err(libc::EROFS));
        assert_eq!(policy.check(FsOpKind::Create, "ws", "/f"), Err(libc::EROFS));
        assert_eq!(policy.check(FsOpKind::Unlink, "ws", "/f"), Err(libc::EROFS));
    }
}
