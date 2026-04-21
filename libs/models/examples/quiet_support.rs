use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;

pub struct QuietStderrGuard {
    saved_stderr_fd: i32,
    _devnull: File,
}

impl QuietStderrGuard {
    pub fn maybe_enable(quiet: bool) -> io::Result<Option<Self>> {
        if quiet {
            Self::enable().map(Some)
        } else {
            Ok(None)
        }
    }

    fn enable() -> io::Result<Self> {
        let devnull = File::options().write(true).open("/dev/null")?;
        let stderr_fd = std::io::stderr().as_raw_fd();

        let saved_stderr_fd = unsafe { libc::dup(stderr_fd) };
        if saved_stderr_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        if unsafe { libc::dup2(devnull.as_raw_fd(), stderr_fd) } < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(saved_stderr_fd);
            }
            return Err(error);
        }

        Ok(Self {
            saved_stderr_fd,
            _devnull: devnull,
        })
    }
}

impl Drop for QuietStderrGuard {
    fn drop(&mut self) {
        let stderr_fd = std::io::stderr().as_raw_fd();
        unsafe {
            libc::dup2(self.saved_stderr_fd, stderr_fd);
            libc::close(self.saved_stderr_fd);
        }
    }
}
