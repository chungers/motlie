use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use crate::types::{HostKeyPolicy, TransferOptions, TmuxSocket};

// ---------------------------------------------------------------------------
// Static-dispatch transport (DC6)
// ---------------------------------------------------------------------------

/// Static-dispatch transport for command execution (DC6).
pub enum TransportKind {
    Local(LocalTransport),
    Mock(MockTransport),
    Ssh(SshTransport),
}

impl TransportKind {
    /// Execute a command and return stdout.
    pub async fn exec(&self, command: &str) -> Result<String> {
        match self {
            TransportKind::Local(t) => t.exec(command).await,
            TransportKind::Mock(t) => t.exec(command).await,
            TransportKind::Ssh(t) => t.exec(command).await,
        }
    }

    /// Transport-agnostic health probe.
    ///
    /// - `Local`: always returns `true` (subprocess transport has no persistent connection).
    /// - `Mock`: always returns `true`.
    /// - `Ssh`: checks if the underlying SSH connection is closed via `SshTransport::is_closed()`.
    pub fn is_healthy(&self) -> bool {
        match self {
            TransportKind::Local(_) => true,
            TransportKind::Mock(_) => true,
            TransportKind::Ssh(t) => !t.is_closed(),
        }
    }

    /// Upload a file or directory from a local path to a remote path (DC23).
    ///
    /// Directory placement follows `cp -r` semantics: if the destination exists
    /// as a directory, the source is copied **into** it; if it doesn't exist,
    /// the source is copied **as** that path. Returns `Result<()>` initially.
    pub async fn upload(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        match self {
            TransportKind::Local(t) => t.upload(local_path, remote_path, opts).await,
            TransportKind::Mock(t) => t.upload(local_path, remote_path, opts).await,
            TransportKind::Ssh(t) => t.upload(local_path, remote_path, opts).await,
        }
    }

    /// Download a file or directory from a remote path to a local path (DC23).
    ///
    /// Directory placement follows `cp -r` semantics: if the destination exists
    /// as a directory, the source is copied **into** it; if it doesn't exist,
    /// the source is copied **as** that path. Returns `Result<()>` initially.
    pub async fn download(
        &self,
        remote_path: &Path,
        local_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        match self {
            TransportKind::Local(t) => t.download(remote_path, local_path, opts).await,
            TransportKind::Mock(t) => t.download(remote_path, local_path, opts).await,
            TransportKind::Ssh(t) => t.download(remote_path, local_path, opts).await,
        }
    }

    /// Open a persistent shell channel.
    ///
    /// `cols` and `rows` control PTY dimensions for SSH transports (used by
    /// `SshTransport::open_shell()`). Local and Mock transports ignore them.
    pub async fn open_shell(&self, cols: u32, rows: u32) -> Result<ShellChannelKind> {
        match self {
            TransportKind::Local(t) => t.open_shell().await.map(ShellChannelKind::Local),
            TransportKind::Mock(t) => t.open_shell().await.map(ShellChannelKind::Mock),
            TransportKind::Ssh(t) => t.open_shell(cols, rows).await.map(ShellChannelKind::Ssh),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared file-transfer helpers (DC23)
// ---------------------------------------------------------------------------

/// Resolve the effective destination path following `cp -r` semantics (DC23).
///
/// - If `dst` exists and is a directory, returns `dst / src.file_name()` (copy into).
/// - If `dst` does not exist, returns `dst` as-is (copy as).
/// - If `src` has no filename component (e.g. `/`), returns an error.
fn resolve_destination(src: &Path, dst: &Path) -> Result<std::path::PathBuf> {
    if dst.is_dir() {
        let basename = src
            .file_name()
            .ok_or_else(|| anyhow!("source path has no filename: {}", src.display()))?;
        Ok(dst.join(basename))
    } else {
        Ok(dst.to_path_buf())
    }
}

/// Copy a file or directory tree on the local filesystem (DC23).
///
/// Implements the full `TransferOptions` contract:
/// - `overwrite=false` + destination exists → `Err`
/// - `recursive=false` + source is directory → `Err`
/// - Symlinks encountered → `Err`
/// - Directory merge semantics when `overwrite=true`
/// - `cp -r` placement (copy into existing dir, copy as missing path)
fn copy_local(src: &Path, dst: &Path, opts: &TransferOptions) -> Result<()> {
    // Validate source exists
    let src_meta = std::fs::symlink_metadata(src)
        .map_err(|e| anyhow!("source not found: {}: {}", src.display(), e))?;

    // Reject symlinks
    if src_meta.file_type().is_symlink() {
        return Err(anyhow!(
            "symlink encountered at source: {}",
            src.display()
        ));
    }

    if src_meta.is_dir() {
        if !opts.recursive {
            return Err(anyhow!(
                "source is a directory but recursive=false: {}",
                src.display()
            ));
        }
        copy_dir_local(src, dst, opts)
    } else {
        copy_file_local(src, dst, opts)
    }
}

/// Copy a single file with overwrite checking.
fn copy_file_local(src: &Path, dst: &Path, opts: &TransferOptions) -> Result<()> {
    // For files, resolve destination: if dst is an existing dir, copy into it
    let effective_dst = resolve_destination(src, dst)?;

    if effective_dst.exists() {
        if !opts.overwrite {
            return Err(anyhow!(
                "destination already exists and overwrite=false: {}",
                effective_dst.display()
            ));
        }
        // Type mismatch: destination is a directory but source is a file
        if effective_dst.is_dir() {
            return Err(anyhow!(
                "type mismatch: source is a file but destination is a directory: {}",
                effective_dst.display()
            ));
        }
    }

    // Ensure parent directory exists
    if let Some(parent) = effective_dst.parent() {
        if !parent.exists() {
            return Err(anyhow!(
                "parent directory does not exist: {}",
                parent.display()
            ));
        }
    }

    std::fs::copy(src, &effective_dst).map_err(|e| {
        anyhow!(
            "failed to copy {} -> {}: {}",
            src.display(),
            effective_dst.display(),
            e
        )
    })?;
    Ok(())
}

/// Copy a directory tree with merge semantics.
fn copy_dir_local(src: &Path, dst: &Path, opts: &TransferOptions) -> Result<()> {
    // Resolve destination using cp -r semantics
    let effective_dst = resolve_destination(src, dst)?;

    if effective_dst.exists() {
        if !opts.overwrite {
            return Err(anyhow!(
                "destination already exists and overwrite=false: {}",
                effective_dst.display()
            ));
        }
        // Type mismatch: destination is a file but source is a directory
        if effective_dst.is_file() {
            return Err(anyhow!(
                "type mismatch: source is a directory but destination is a file: {}",
                effective_dst.display()
            ));
        }
    } else {
        // Ensure parent of effective_dst exists
        if let Some(parent) = effective_dst.parent() {
            if !parent.exists() {
                return Err(anyhow!(
                    "parent directory does not exist: {}",
                    parent.display()
                ));
            }
        }
        std::fs::create_dir(&effective_dst).map_err(|e| {
            anyhow!(
                "failed to create directory {}: {}",
                effective_dst.display(),
                e
            )
        })?;
    }

    // Recursively copy contents
    copy_dir_contents(src, &effective_dst, opts)
}

/// Recursively copy directory contents with merge semantics.
fn copy_dir_contents(src: &Path, dst: &Path, opts: &TransferOptions) -> Result<()> {
    for entry in std::fs::read_dir(src)
        .map_err(|e| anyhow!("failed to read directory {}: {}", src.display(), e))?
    {
        let entry =
            entry.map_err(|e| anyhow!("failed to read dir entry in {}: {}", src.display(), e))?;
        let entry_path = entry.path();
        let entry_meta = std::fs::symlink_metadata(&entry_path).map_err(|e| {
            anyhow!(
                "failed to stat {}: {}",
                entry_path.display(),
                e
            )
        })?;

        // Reject symlinks
        if entry_meta.file_type().is_symlink() {
            return Err(anyhow!(
                "symlink encountered: {}",
                entry_path.display()
            ));
        }

        let name = entry.file_name();
        let dst_entry = dst.join(&name);

        if entry_meta.is_dir() {
            if dst_entry.exists() {
                if dst_entry.is_file() {
                    return Err(anyhow!(
                        "type mismatch: source is a directory but destination is a file: {}",
                        dst_entry.display()
                    ));
                }
                // Merge into existing directory
            } else {
                std::fs::create_dir(&dst_entry).map_err(|e| {
                    anyhow!(
                        "failed to create directory {}: {}",
                        dst_entry.display(),
                        e
                    )
                })?;
            }
            copy_dir_contents(&entry_path, &dst_entry, opts)?;
        } else {
            // Regular file
            if dst_entry.exists() {
                if dst_entry.is_dir() {
                    return Err(anyhow!(
                        "type mismatch: source is a file but destination is a directory: {}",
                        dst_entry.display()
                    ));
                }
                // overwrite=true is implied here since we passed the top-level check
            }
            std::fs::copy(&entry_path, &dst_entry).map_err(|e| {
                anyhow!(
                    "failed to copy {} -> {}: {}",
                    entry_path.display(),
                    dst_entry.display(),
                    e
                )
            })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LocalTransport
// ---------------------------------------------------------------------------

/// Localhost transport — executes via subprocess.
pub struct LocalTransport {
    pub timeout: std::time::Duration,
}

impl LocalTransport {
    pub fn new() -> Self {
        LocalTransport {
            timeout: std::time::Duration::from_secs(10),
        }
    }

    pub fn with_timeout(timeout: std::time::Duration) -> Self {
        LocalTransport { timeout }
    }

    async fn exec(&self, command: &str) -> Result<String> {
        let output = tokio::time::timeout(
            self.timeout,
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| anyhow!("command timed out after {:?}: {}", self.timeout, command))?
        .map_err(|e| anyhow!("failed to execute command: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "command failed (exit {}): {}\nstderr: {}",
                output.status.code().unwrap_or(-1),
                command,
                stderr.trim()
            ));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn open_shell(&self) -> Result<LocalShellChannel> {
        let child = Command::new("sh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("failed to spawn shell: {}", e))?;

        Ok(LocalShellChannel { child })
    }

    /// Upload: local filesystem copy (local_path → remote_path).
    /// For LocalTransport, both paths are on the same machine.
    async fn upload(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        tokio::time::timeout(self.timeout, async {
            copy_local(local_path, remote_path, opts)
        })
        .await
        .map_err(|_| {
            anyhow!(
                "upload timed out after {:?}: {} -> {}",
                self.timeout,
                local_path.display(),
                remote_path.display()
            )
        })?
    }

    /// Download: local filesystem copy (remote_path → local_path).
    /// For LocalTransport, both paths are on the same machine.
    async fn download(
        &self,
        remote_path: &Path,
        local_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        tokio::time::timeout(self.timeout, async {
            copy_local(remote_path, local_path, opts)
        })
        .await
        .map_err(|_| {
            anyhow!(
                "download timed out after {:?}: {} -> {}",
                self.timeout,
                remote_path.display(),
                local_path.display()
            )
        })?
    }
}

impl Default for LocalTransport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MockTransport
// ---------------------------------------------------------------------------

/// In-memory filesystem entry for MockTransport (DC23).
#[derive(Debug, Clone)]
pub enum MockFsEntry {
    File(Vec<u8>),
    Dir,
}

/// Mock transport for unit testing.
///
/// Patterns are matched in insertion order (first match wins), which makes
/// test behavior deterministic regardless of hash seed. Use `with_response()`
/// to add success responses and `with_error()` to add error responses.
///
/// For file transfer testing (DC23), an in-memory filesystem is available
/// via `with_file()`, `with_dir()`, and accessor methods.
pub struct MockTransport {
    /// Ordered (pattern, response_queue) pairs. First-match wins.
    responses: Mutex<Vec<(String, Vec<String>)>>,
    /// Ordered (pattern, error_message) pairs. Checked before `responses`.
    errors: Mutex<Vec<(String, String)>>,
    default_response: String,
    /// In-memory filesystem for upload/download testing (DC23).
    fs: Mutex<HashMap<std::path::PathBuf, MockFsEntry>>,
    /// Optional transfer error injection.
    transfer_error: Mutex<Option<String>>,
}

impl MockTransport {
    pub fn new() -> Self {
        MockTransport {
            responses: Mutex::new(Vec::new()),
            errors: Mutex::new(Vec::new()),
            default_response: String::new(),
            fs: Mutex::new(HashMap::new()),
            transfer_error: Mutex::new(None),
        }
    }

    /// Add a canned response for a command pattern.
    /// If multiple responses are added for the same pattern, they are
    /// returned in FIFO order. When exhausted, returns the last one.
    /// Patterns are matched in insertion order (first match wins).
    pub fn with_response(self, command_contains: &str, response: &str) -> Self {
        let mut responses = self.responses.lock().unwrap();
        if let Some(entry) = responses.iter_mut().find(|(p, _)| p == command_contains) {
            entry.1.push(response.to_string());
        } else {
            responses.push((command_contains.to_string(), vec![response.to_string()]));
        }
        drop(responses);
        self
    }

    /// Add a canned error for a command pattern.
    /// When `exec()` matches this pattern, it returns `Err(anyhow!(...))`.
    /// Error patterns are checked before response patterns.
    pub fn with_error(self, command_contains: &str, message: &str) -> Self {
        let mut errors = self.errors.lock().unwrap();
        errors.push((command_contains.to_string(), message.to_string()));
        drop(errors);
        self
    }

    /// Set default response for unmatched commands.
    pub fn with_default(mut self, response: &str) -> Self {
        self.default_response = response.to_string();
        self
    }

    async fn exec(&self, command: &str) -> Result<String> {
        // Check error patterns first (insertion order)
        {
            let errors = self.errors.lock().unwrap();
            for (pattern, message) in errors.iter() {
                if command.contains(pattern.as_str()) {
                    return Err(anyhow!("{}", message));
                }
            }
        }

        // Check response patterns (insertion order, first match wins)
        let mut responses = self.responses.lock().unwrap();
        for (pattern, queue) in responses.iter_mut() {
            if command.contains(pattern.as_str()) {
                if queue.len() > 1 {
                    return Ok(queue.remove(0));
                } else {
                    return Ok(queue[0].clone());
                }
            }
        }
        Ok(self.default_response.clone())
    }

    async fn open_shell(&self) -> Result<MockShellChannel> {
        Ok(MockShellChannel {
            data: Vec::new(),
            pos: 0,
        })
    }

    // --- In-memory filesystem for transfer testing (DC23) ---

    /// Pre-populate the mock filesystem with a file.
    pub fn with_file(self, path: impl Into<std::path::PathBuf>, contents: impl Into<Vec<u8>>) -> Self {
        let path = path.into();
        // Ensure parent dirs exist
        let mut fs = self.fs.lock().unwrap();
        for ancestor in path.ancestors().skip(1) {
            if ancestor == Path::new("") || ancestor == Path::new("/") {
                continue;
            }
            fs.entry(ancestor.to_path_buf())
                .or_insert(MockFsEntry::Dir);
        }
        fs.insert(path, MockFsEntry::File(contents.into()));
        drop(fs);
        self
    }

    /// Pre-populate the mock filesystem with an empty directory.
    pub fn with_dir(self, path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        let mut fs = self.fs.lock().unwrap();
        for ancestor in path.ancestors().skip(1) {
            if ancestor == Path::new("") || ancestor == Path::new("/") {
                continue;
            }
            fs.entry(ancestor.to_path_buf())
                .or_insert(MockFsEntry::Dir);
        }
        fs.insert(path, MockFsEntry::Dir);
        drop(fs);
        self
    }

    /// Inject a transfer error that will be returned on the next upload/download.
    pub fn with_transfer_error(self, message: &str) -> Self {
        *self.transfer_error.lock().unwrap() = Some(message.to_string());
        self
    }

    /// Read a file from the mock filesystem.
    pub fn read_file(&self, path: &Path) -> Option<Vec<u8>> {
        let fs = self.fs.lock().unwrap();
        match fs.get(path) {
            Some(MockFsEntry::File(data)) => Some(data.clone()),
            _ => None,
        }
    }

    /// Check if a path exists in the mock filesystem.
    pub fn exists(&self, path: &Path) -> bool {
        self.fs.lock().unwrap().contains_key(path)
    }

    /// List entries in a mock directory.
    pub fn list_dir(&self, path: &Path) -> Vec<std::path::PathBuf> {
        let fs = self.fs.lock().unwrap();
        fs.keys()
            .filter(|k| k.parent() == Some(path) && *k != path)
            .cloned()
            .collect()
    }

    /// Upload: copy from real filesystem into mock filesystem.
    async fn upload(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        // Check for injected error
        if let Some(msg) = self.transfer_error.lock().unwrap().take() {
            return Err(anyhow!("{}", msg));
        }

        let src_meta = std::fs::symlink_metadata(local_path)
            .map_err(|e| anyhow!("source not found: {}: {}", local_path.display(), e))?;

        if src_meta.file_type().is_symlink() {
            return Err(anyhow!("symlink encountered at source: {}", local_path.display()));
        }

        if src_meta.is_dir() {
            if !opts.recursive {
                return Err(anyhow!(
                    "source is a directory but recursive=false: {}",
                    local_path.display()
                ));
            }
            self.mock_upload_dir(local_path, remote_path, opts)
        } else {
            self.mock_upload_file(local_path, remote_path, opts)
        }
    }

    fn mock_upload_file(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        let mut fs = self.fs.lock().unwrap();
        let effective_dst = if fs.get(remote_path).map_or(false, |e| matches!(e, MockFsEntry::Dir)) {
            let name = local_path.file_name()
                .ok_or_else(|| anyhow!("source has no filename: {}", local_path.display()))?;
            remote_path.join(name)
        } else {
            remote_path.to_path_buf()
        };

        if let Some(existing) = fs.get(&effective_dst) {
            if !opts.overwrite {
                return Err(anyhow!(
                    "destination already exists and overwrite=false: {}",
                    effective_dst.display()
                ));
            }
            if matches!(existing, MockFsEntry::Dir) {
                return Err(anyhow!(
                    "type mismatch: source is a file but destination is a directory: {}",
                    effective_dst.display()
                ));
            }
        }

        // Check parent exists
        if let Some(parent) = effective_dst.parent() {
            if parent != Path::new("") && parent != Path::new("/") && !fs.contains_key(parent) {
                return Err(anyhow!(
                    "parent directory does not exist: {}",
                    parent.display()
                ));
            }
        }

        let data = std::fs::read(local_path)
            .map_err(|e| anyhow!("failed to read {}: {}", local_path.display(), e))?;
        fs.insert(effective_dst, MockFsEntry::File(data));
        Ok(())
    }

    fn mock_upload_dir(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        let effective_dst = {
            let fs = self.fs.lock().unwrap();
            if fs.get(remote_path).map_or(false, |e| matches!(e, MockFsEntry::Dir)) {
                let name = local_path.file_name()
                    .ok_or_else(|| anyhow!("source has no filename: {}", local_path.display()))?;
                remote_path.join(name)
            } else {
                remote_path.to_path_buf()
            }
        };

        {
            let mut fs = self.fs.lock().unwrap();
            if let Some(existing) = fs.get(&effective_dst) {
                if !opts.overwrite {
                    return Err(anyhow!(
                        "destination already exists and overwrite=false: {}",
                        effective_dst.display()
                    ));
                }
                if matches!(existing, MockFsEntry::File(_)) {
                    return Err(anyhow!(
                        "type mismatch: source is a directory but destination is a file: {}",
                        effective_dst.display()
                    ));
                }
            } else {
                // Check parent exists
                if let Some(parent) = effective_dst.parent() {
                    if parent != Path::new("") && parent != Path::new("/") && !fs.contains_key(parent) {
                        return Err(anyhow!(
                            "parent directory does not exist: {}",
                            parent.display()
                        ));
                    }
                }
                fs.insert(effective_dst.clone(), MockFsEntry::Dir);
            }
        }

        self.mock_upload_dir_contents(local_path, &effective_dst, opts)
    }

    fn mock_upload_dir_contents(
        &self,
        local_path: &Path,
        remote_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        for entry in std::fs::read_dir(local_path)
            .map_err(|e| anyhow!("failed to read dir {}: {}", local_path.display(), e))?
        {
            let entry = entry.map_err(|e| anyhow!("dir entry error: {}", e))?;
            let entry_path = entry.path();
            let meta = std::fs::symlink_metadata(&entry_path)
                .map_err(|e| anyhow!("failed to stat {}: {}", entry_path.display(), e))?;

            if meta.file_type().is_symlink() {
                return Err(anyhow!("symlink encountered: {}", entry_path.display()));
            }

            let name = entry.file_name();
            let remote_entry = remote_path.join(&name);

            if meta.is_dir() {
                {
                    let mut fs = self.fs.lock().unwrap();
                    if !fs.contains_key(&remote_entry) {
                        fs.insert(remote_entry.clone(), MockFsEntry::Dir);
                    }
                }
                self.mock_upload_dir_contents(&entry_path, &remote_entry, opts)?;
            } else {
                let data = std::fs::read(&entry_path)
                    .map_err(|e| anyhow!("failed to read {}: {}", entry_path.display(), e))?;
                let mut fs = self.fs.lock().unwrap();
                fs.insert(remote_entry, MockFsEntry::File(data));
            }
        }
        Ok(())
    }

    /// Download: copy from mock filesystem to real filesystem.
    async fn download(
        &self,
        remote_path: &Path,
        local_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        // Check for injected error
        if let Some(msg) = self.transfer_error.lock().unwrap().take() {
            return Err(anyhow!("{}", msg));
        }

        let fs = self.fs.lock().unwrap();
        let entry = fs.get(remote_path)
            .ok_or_else(|| anyhow!("source not found: {}", remote_path.display()))?
            .clone();
        drop(fs);

        match entry {
            MockFsEntry::Dir => {
                if !opts.recursive {
                    return Err(anyhow!(
                        "source is a directory but recursive=false: {}",
                        remote_path.display()
                    ));
                }
                self.mock_download_dir(remote_path, local_path, opts)
            }
            MockFsEntry::File(data) => {
                self.mock_download_file(remote_path, &data, local_path, opts)
            }
        }
    }

    fn mock_download_file(
        &self,
        remote_path: &Path,
        data: &[u8],
        local_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        let effective_dst = if local_path.is_dir() {
            let name = remote_path.file_name()
                .ok_or_else(|| anyhow!("source has no filename: {}", remote_path.display()))?;
            local_path.join(name)
        } else {
            local_path.to_path_buf()
        };

        if effective_dst.exists() {
            if !opts.overwrite {
                return Err(anyhow!(
                    "destination already exists and overwrite=false: {}",
                    effective_dst.display()
                ));
            }
            if effective_dst.is_dir() {
                return Err(anyhow!(
                    "type mismatch: source is a file but destination is a directory: {}",
                    effective_dst.display()
                ));
            }
        }

        if let Some(parent) = effective_dst.parent() {
            if !parent.exists() {
                return Err(anyhow!(
                    "parent directory does not exist: {}",
                    parent.display()
                ));
            }
        }

        std::fs::write(&effective_dst, data)
            .map_err(|e| anyhow!("failed to write {}: {}", effective_dst.display(), e))?;
        Ok(())
    }

    fn mock_download_dir(
        &self,
        remote_path: &Path,
        local_path: &Path,
        opts: &TransferOptions,
    ) -> Result<()> {
        let effective_dst = if local_path.is_dir() {
            let name = remote_path.file_name()
                .ok_or_else(|| anyhow!("source has no filename: {}", remote_path.display()))?;
            local_path.join(name)
        } else {
            local_path.to_path_buf()
        };

        if effective_dst.exists() {
            if !opts.overwrite {
                return Err(anyhow!(
                    "destination already exists and overwrite=false: {}",
                    effective_dst.display()
                ));
            }
            if effective_dst.is_file() {
                return Err(anyhow!(
                    "type mismatch: source is a directory but destination is a file: {}",
                    effective_dst.display()
                ));
            }
        } else {
            if let Some(parent) = effective_dst.parent() {
                if !parent.exists() {
                    return Err(anyhow!(
                        "parent directory does not exist: {}",
                        parent.display()
                    ));
                }
            }
            std::fs::create_dir(&effective_dst)
                .map_err(|e| anyhow!("failed to create dir {}: {}", effective_dst.display(), e))?;
        }

        // Download children from mock fs
        let children: Vec<(std::path::PathBuf, MockFsEntry)> = {
            let fs = self.fs.lock().unwrap();
            fs.iter()
                .filter(|(k, _)| k.parent() == Some(remote_path) && *k != remote_path)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        for (child_path, entry) in children {
            let name = child_path.file_name()
                .ok_or_else(|| anyhow!("child has no filename: {}", child_path.display()))?;
            let local_child = effective_dst.join(name);
            match entry {
                MockFsEntry::File(data) => {
                    std::fs::write(&local_child, &data)
                        .map_err(|e| anyhow!("failed to write {}: {}", local_child.display(), e))?;
                }
                MockFsEntry::Dir => {
                    if !local_child.exists() {
                        std::fs::create_dir(&local_child)
                            .map_err(|e| anyhow!("failed to create dir {}: {}", local_child.display(), e))?;
                    }
                    self.mock_download_dir(&child_path, &local_child, &TransferOptions {
                        overwrite: opts.overwrite,
                        recursive: true,
                    })?;
                }
            }
        }
        Ok(())
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SshTransport (Phase 2a.1, DC2, DC6)
// ---------------------------------------------------------------------------

/// SSH/host connection configuration.
///
/// Constructed via builder (`SshConfig::new()`) or parsed from an SSH URI
/// string (`SshConfig::parse()`). Supports both nassh-style (`;` in userinfo)
/// and query-param (`?key=value`) syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshConfig {
    host: String,
    port: u16,
    user: String,
    host_key_policy: HostKeyPolicy,
    timeout: std::time::Duration,
    keepalive_interval: Option<std::time::Duration>,
    socket: Option<TmuxSocket>,
}

impl SshConfig {
    // --- Builder ---

    pub fn new(host: impl Into<String>, user: impl Into<String>) -> Self {
        SshConfig {
            host: host.into(),
            port: 22,
            user: user.into(),
            host_key_policy: HostKeyPolicy::default(),
            timeout: std::time::Duration::from_secs(10),
            keepalive_interval: Some(std::time::Duration::from_secs(30)),
            socket: None,
        }
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_host_key_policy(mut self, policy: HostKeyPolicy) -> Self {
        self.host_key_policy = policy;
        self
    }

    /// Set the per-command execution timeout for the SSH transport.
    ///
    /// This bounds each individual `exec()` call (channel open + exec + output
    /// collection). It is independent of `Target::exec()`'s sentinel-poll
    /// timeout, which bounds how long to wait for a user command's output
    /// to appear in scrollback. Keep this short (seconds) to catch hung
    /// connections; use a larger `Target::exec()` timeout for long-running
    /// user commands.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_keepalive(mut self, interval: Option<std::time::Duration>) -> Self {
        self.keepalive_interval = interval;
        self
    }

    /// Set the tmux socket for this config.
    ///
    /// `TmuxSocket::Name` values are restricted to `[A-Za-z0-9._-]+` for
    /// URI round-trip safety and tmux compatibility. Returns `Err` on
    /// invalid names. `TmuxSocket::Path` values are not restricted
    /// (filesystem paths).
    pub fn with_socket(mut self, socket: TmuxSocket) -> anyhow::Result<Self> {
        if let TmuxSocket::Name(ref name) = socket {
            if !is_valid_socket_name(name) {
                return Err(anyhow::anyhow!(
                    "invalid socket name '{}': must match [A-Za-z0-9._-]+",
                    name
                ));
            }
        }
        self.socket = Some(socket);
        Ok(self)
    }

    // --- Accessors ---

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn user(&self) -> &str {
        &self.user
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn host_key_policy(&self) -> &HostKeyPolicy {
        &self.host_key_policy
    }

    pub fn timeout(&self) -> std::time::Duration {
        self.timeout
    }

    pub fn keepalive_interval(&self) -> Option<std::time::Duration> {
        self.keepalive_interval
    }

    pub fn socket(&self) -> Option<&TmuxSocket> {
        self.socket.as_ref()
    }

    pub fn is_localhost(&self) -> bool {
        self.host == "localhost" || self.host == "127.0.0.1" || self.host == "::1"
    }
}

/// SSH client handler implementing host key verification (DC2).
struct SshHandler {
    host: String,
    port: u16,
    policy: HostKeyPolicy,
}

#[async_trait::async_trait]
impl russh::client::Handler for SshHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh_keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        match &self.policy {
            HostKeyPolicy::Verify => {
                match russh_keys::known_hosts::check_known_hosts(&self.host, self.port, server_public_key) {
                    Ok(true) => Ok(true),
                    Ok(false) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            "SSH host key not found in known_hosts. \
                             Add the key with: ssh-keyscan -p {} {} >> ~/.ssh/known_hosts",
                            self.port,
                            self.host
                        );
                        Ok(false)
                    }
                    Err(russh_keys::Error::KeyChanged { line }) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            line,
                            "SSH HOST KEY HAS CHANGED — possible MITM attack. \
                             The key at ~/.ssh/known_hosts line {} does not match. \
                             If this is expected, remove the old entry and reconnect.",
                            line
                        );
                        Ok(false)
                    }
                    Err(e) => {
                        tracing::error!(
                            host = %self.host,
                            error = %e,
                            "Failed to check known_hosts"
                        );
                        Ok(false)
                    }
                }
            }
            HostKeyPolicy::TrustFirstUse => {
                match russh_keys::known_hosts::check_known_hosts(&self.host, self.port, server_public_key) {
                    Ok(true) => Ok(true),
                    Ok(false) => {
                        // First connection — learn the key
                        tracing::info!(
                            host = %self.host,
                            port = self.port,
                            "Trust-on-first-use: accepting and persisting new host key"
                        );
                        if let Err(e) = russh_keys::known_hosts::learn_known_hosts(
                            &self.host,
                            self.port,
                            server_public_key,
                        ) {
                            tracing::error!(
                                host = %self.host,
                                error = %e,
                                "TOFU: failed to persist host key — rejecting connection. \
                                 Check that ~/.ssh/known_hosts is writable."
                            );
                            return Ok(false);
                        }
                        Ok(true)
                    }
                    Err(russh_keys::Error::KeyChanged { line }) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            line,
                            "SSH HOST KEY HAS CHANGED — rejecting (TOFU policy). \
                             The key at ~/.ssh/known_hosts line {} does not match.",
                            line
                        );
                        Ok(false)
                    }
                    Err(e) => {
                        tracing::warn!(
                            host = %self.host,
                            error = %e,
                            "Failed to check known_hosts, attempting to learn key (TOFU)"
                        );
                        if let Err(e) = russh_keys::known_hosts::learn_known_hosts(
                            &self.host,
                            self.port,
                            server_public_key,
                        ) {
                            tracing::error!(
                                error = %e,
                                "TOFU: failed to persist host key — rejecting connection"
                            );
                            return Ok(false);
                        }
                        Ok(true)
                    }
                }
            }
            HostKeyPolicy::Insecure => {
                tracing::warn!(
                    host = %self.host,
                    port = self.port,
                    "Insecure mode: accepting SSH host key without verification"
                );
                Ok(true)
            }
        }
    }
}

/// SSH transport — executes commands on a remote host via russh (Phase 2a.1).
pub struct SshTransport {
    handle: Arc<tokio::sync::Mutex<russh::client::Handle<SshHandler>>>,
    config: SshConfig,
}

impl SshTransport {
    /// Connect to a remote host via SSH and authenticate using ssh-agent.
    ///
    /// Returns an error with actionable message if SSH_AUTH_SOCK is not set
    /// or the agent has no identities (OC3).
    pub async fn connect(config: SshConfig) -> Result<Self> {
        let ssh_config = russh::client::Config {
            inactivity_timeout: Some(config.timeout),
            keepalive_interval: config.keepalive_interval,
            ..<_>::default()
        };

        let handler = SshHandler {
            host: config.host.clone(),
            port: config.port,
            policy: config.host_key_policy.clone(),
        };

        let addr = format!("{}:{}", config.host, config.port);
        let mut handle = tokio::time::timeout(
            config.timeout,
            russh::client::connect(Arc::new(ssh_config), &addr, handler),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "SSH connection to {}:{} timed out after {:?}",
                config.host,
                config.port,
                config.timeout
            )
        })?
        .map_err(|e| {
            anyhow!(
                "SSH connection to {}:{} failed: {}",
                config.host,
                config.port,
                e
            )
        })?;

        // Authenticate via ssh-agent
        Self::authenticate_agent(&mut handle, &config).await?;

        Ok(SshTransport {
            handle: Arc::new(tokio::sync::Mutex::new(handle)),
            config,
        })
    }

    /// Authenticate using ssh-agent keys.
    async fn authenticate_agent(
        handle: &mut russh::client::Handle<SshHandler>,
        config: &SshConfig,
    ) -> Result<()> {
        let mut agent = russh_keys::agent::client::AgentClient::connect_env()
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to connect to SSH agent (is SSH_AUTH_SOCK set?): {}. \
                     Ensure ssh-agent is running and SSH_AUTH_SOCK is exported.",
                    e
                )
            })?;

        let identities = agent.request_identities().await.map_err(|e| {
            anyhow!(
                "Failed to list SSH agent identities: {}. \
                 Ensure keys are loaded with ssh-add.",
                e
            )
        })?;

        if identities.is_empty() {
            return Err(anyhow!(
                "SSH agent has no identities. Add a key with: ssh-add ~/.ssh/id_ed25519"
            ));
        }

        // Try each agent key until one succeeds
        for key in &identities {
            let (returned_agent, auth_result) = handle
                .authenticate_future(&config.user, key.clone(), agent)
                .await;
            agent = returned_agent;

            match auth_result {
                Ok(true) => {
                    tracing::debug!(
                        host = %config.host,
                        user = %config.user,
                        "SSH authentication succeeded"
                    );
                    return Ok(());
                }
                Ok(false) => {
                    tracing::debug!("SSH key rejected, trying next identity");
                    continue;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "SSH auth error with key, trying next");
                    continue;
                }
            }
        }

        Err(anyhow!(
            "SSH authentication failed for user '{}' on {}:{}. \
             None of the {} agent key(s) were accepted.",
            config.user,
            config.host,
            config.port,
            identities.len()
        ))
    }

    /// Execute a command on the remote host and return stdout.
    ///
    /// The full exec lifecycle — channel open, exec request, and output
    /// collection — is bounded by `config.timeout`. The SSH handle lock is
    /// held only during channel open (not the full command lifetime), allowing
    /// multiple concurrent execs on the same connection.
    async fn exec(&self, command: &str) -> Result<String> {
        // Single timeout boundary covering channel open + exec + output
        // collection, so a stalled server at any phase is caught.
        let (stdout, stderr, exit_code) = tokio::time::timeout(
            self.config.timeout,
            async {
                // Lock only to open the channel, then release. The Channel is
                // self-contained — its read/write operations don't need the Handle.
                let mut channel = {
                    let handle = self.handle.lock().await;
                    handle.channel_open_session().await.map_err(|e| {
                        anyhow!("SSH: failed to open session channel: {}", e)
                    })?
                };

                channel.exec(true, command).await.map_err(|e| {
                    anyhow!("SSH: failed to exec command: {}", e)
                })?;

                // Collect output
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                let mut exit_code: Option<u32> = None;

                while let Some(msg) = channel.wait().await {
                    match msg {
                        russh::ChannelMsg::Data { ref data } => {
                            stdout.extend_from_slice(data);
                        }
                        russh::ChannelMsg::ExtendedData { ref data, ext } => {
                            if ext == 1 {
                                stderr.extend_from_slice(data);
                            }
                        }
                        russh::ChannelMsg::ExitStatus { exit_status } => {
                            exit_code = Some(exit_status);
                        }
                        russh::ChannelMsg::Eof | russh::ChannelMsg::Close => break,
                        _ => {}
                    }
                }

                Ok::<_, anyhow::Error>((stdout, stderr, exit_code))
            },
        )
        .await
        .map_err(|_| {
            anyhow!(
                "SSH command timed out after {:?}: {}",
                self.config.timeout,
                command
            )
        })??;

        let code = exit_code.unwrap_or(0);
        if code != 0 {
            let stderr_str = String::from_utf8_lossy(&stderr);
            return Err(anyhow!(
                "command failed (exit {}): {}\nstderr: {}",
                code,
                command,
                stderr_str.trim()
            ));
        }

        Ok(String::from_utf8_lossy(&stdout).to_string())
    }

    /// Open a persistent shell channel on the remote host with a PTY.
    ///
    /// `cols` and `rows` set the initial PTY dimensions. Use the target pane's
    /// geometry for accurate rendering, or pass `(80, 24)` as a safe default.
    async fn open_shell(&self, cols: u32, rows: u32) -> Result<SshShellChannel> {
        let channel = {
            let handle = self.handle.lock().await;
            handle.channel_open_session().await.map_err(|e| {
                anyhow!("SSH: failed to open session channel for shell: {}", e)
            })?
        };

        // Request a PTY for interactive shell use
        channel
            .request_pty(
                true,
                "xterm",
                cols,
                rows,
                0,    // pixel width
                0,    // pixel height
                &[],  // terminal modes
            )
            .await
            .map_err(|e| anyhow!("SSH: failed to request PTY: {}", e))?;

        channel
            .request_shell(true)
            .await
            .map_err(|e| anyhow!("SSH: failed to request shell: {}", e))?;

        Ok(SshShellChannel { channel })
    }

    /// Check if the SSH connection is still alive.
    pub fn is_closed(&self) -> bool {
        // Try to check without blocking — if we can't get the lock, assume alive
        match self.handle.try_lock() {
            Ok(handle) => handle.is_closed(),
            Err(_) => false,
        }
    }

    /// Get a reference to the SSH configuration.
    pub fn config(&self) -> &SshConfig {
        &self.config
    }

    /// Upload a file or directory to the remote host via SFTP (DC23).
    ///
    /// TODO(1.13e): Implement using russh-sftp. Currently returns an error
    /// indicating SFTP is not yet implemented.
    async fn upload(
        &self,
        _local_path: &Path,
        _remote_path: &Path,
        _opts: &TransferOptions,
    ) -> Result<()> {
        Err(anyhow!("SFTP upload not yet implemented for SSH transport"))
    }

    /// Download a file or directory from the remote host via SFTP (DC23).
    ///
    /// TODO(1.13e): Implement using russh-sftp. Currently returns an error
    /// indicating SFTP is not yet implemented.
    async fn download(
        &self,
        _remote_path: &Path,
        _local_path: &Path,
        _opts: &TransferOptions,
    ) -> Result<()> {
        Err(anyhow!("SFTP download not yet implemented for SSH transport"))
    }
}

// ---------------------------------------------------------------------------
// Shell channels
// ---------------------------------------------------------------------------

/// Shell channel — static dispatch.
pub enum ShellChannelKind {
    Local(LocalShellChannel),
    Mock(MockShellChannel),
    Ssh(SshShellChannel),
}

impl ShellChannelKind {
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        match self {
            ShellChannelKind::Local(ch) => ch.write(data).await,
            ShellChannelKind::Mock(ch) => ch.write(data).await,
            ShellChannelKind::Ssh(ch) => ch.write(data).await,
        }
    }

    pub async fn read(&mut self) -> Option<ShellEvent> {
        match self {
            ShellChannelKind::Local(ch) => ch.read().await,
            ShellChannelKind::Mock(ch) => ch.read().await,
            ShellChannelKind::Ssh(ch) => ch.read().await,
        }
    }
}

/// Events from a shell channel.
#[derive(Debug)]
pub enum ShellEvent {
    Data(Vec<u8>),
    Eof,
}

/// Localhost shell channel backed by a child process.
pub struct LocalShellChannel {
    child: Child,
}

impl LocalShellChannel {
    async fn write(&mut self, data: &[u8]) -> Result<()> {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("stdin not available"))?;
        stdin
            .write_all(data)
            .await
            .map_err(|e| anyhow!("write to shell failed: {}", e))
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        let stdout = self.child.stdout.as_mut()?;
        let mut buf = vec![0u8; 4096];
        match stdout.read(&mut buf).await {
            Ok(0) => Some(ShellEvent::Eof),
            Ok(n) => {
                buf.truncate(n);
                Some(ShellEvent::Data(buf))
            }
            Err(_) => Some(ShellEvent::Eof),
        }
    }
}

/// SSH shell channel backed by a russh PTY session.
pub struct SshShellChannel {
    channel: russh::Channel<russh::client::Msg>,
}

impl SshShellChannel {
    async fn write(&mut self, data: &[u8]) -> Result<()> {
        self.channel
            .data(&data[..])
            .await
            .map_err(|e| anyhow!("SSH: write to shell failed: {}", e))
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        match self.channel.wait().await {
            Some(russh::ChannelMsg::Data { data }) => {
                Some(ShellEvent::Data(data.to_vec()))
            }
            Some(russh::ChannelMsg::ExtendedData { data, .. }) => {
                Some(ShellEvent::Data(data.to_vec()))
            }
            Some(russh::ChannelMsg::Eof) | Some(russh::ChannelMsg::Close) | None => {
                Some(ShellEvent::Eof)
            }
            Some(_) => {
                // Other messages (ExitStatus, etc.) — skip and read again
                // Recurse via Box::pin to avoid stack growth
                Box::pin(self.read()).await
            }
        }
    }
}

/// Mock shell channel for testing.
pub struct MockShellChannel {
    data: Vec<Vec<u8>>,
    pos: usize,
}

impl MockShellChannel {
    async fn write(&mut self, _data: &[u8]) -> Result<()> {
        Ok(())
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        if self.pos < self.data.len() {
            let d = self.data[self.pos].clone();
            self.pos += 1;
            Some(ShellEvent::Data(d))
        } else {
            Some(ShellEvent::Eof)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build the tmux command prefix with optional socket args.
pub fn tmux_prefix(socket: Option<&TmuxSocket>) -> String {
    match socket {
        None => "tmux".to_string(),
        Some(TmuxSocket::Name(n)) => format!("tmux -L '{}'", shell_escape_arg(n)),
        Some(TmuxSocket::Path(p)) => format!("tmux -S '{}'", shell_escape_arg(p)),
    }
}

/// Validate a tmux socket name against the allowed character set.
///
/// Socket names must be non-empty and contain only `[A-Za-z0-9._-]`.
/// This restriction ensures URI round-trip safety (no reserved chars)
/// and tmux compatibility.
pub fn is_valid_socket_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// POSIX shell escape: single-quote wrapping with '\'' for interior quotes.
pub fn shell_escape_arg(s: &str) -> String {
    s.replace('\'', "'\\''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_transport_canned_response() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "session1\nsession2\n")
            .with_response("list-panes", "pane1\npane2\n");

        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "session1\nsession2\n");

        let result = mock.exec("tmux list-panes -a").await.unwrap();
        assert_eq!(result, "pane1\npane2\n");
    }

    #[tokio::test]
    async fn mock_transport_default_response() {
        let mock = MockTransport::new().with_default("ok");
        let result = mock.exec("anything").await.unwrap();
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn mock_transport_error_response() {
        let mock = MockTransport::new()
            .with_error("kill-session", "session not found")
            .with_response("list-sessions", "ok\n");
        // Error pattern matches → Err
        let result = mock.exec("tmux kill-session -t foo").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("session not found"));
        // Non-error pattern still works
        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "ok\n");
    }

    #[tokio::test]
    async fn mock_transport_error_before_response() {
        // Error patterns are checked before response patterns
        let mock = MockTransport::new()
            .with_response("cmd", "ok")
            .with_error("cmd", "fail");
        let result = mock.exec("cmd").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn mock_transport_insertion_order() {
        // First-match wins: "list" matches before "list-sessions"
        let mock = MockTransport::new()
            .with_response("list", "first")
            .with_response("list-sessions", "second");
        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "first");
    }

    #[tokio::test]
    async fn local_transport_echo() {
        let local = LocalTransport::new();
        let result = local.exec("echo hello").await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn tmux_prefix_default() {
        assert_eq!(tmux_prefix(None), "tmux");
    }

    #[test]
    fn tmux_prefix_named() {
        let socket = TmuxSocket::Name("myserver".to_string());
        assert_eq!(tmux_prefix(Some(&socket)), "tmux -L 'myserver'");
    }

    #[test]
    fn tmux_prefix_path() {
        let socket = TmuxSocket::Path("/tmp/tmux.sock".to_string());
        assert_eq!(tmux_prefix(Some(&socket)), "tmux -S '/tmp/tmux.sock'");
    }

    #[test]
    fn shell_escape_simple() {
        assert_eq!(shell_escape_arg("hello"), "hello");
    }

    #[test]
    fn shell_escape_quotes() {
        assert_eq!(shell_escape_arg("it's"), "it'\\''s");
    }

    #[test]
    fn shell_escape_multiple_quotes() {
        assert_eq!(shell_escape_arg("a'b'c"), "a'\\''b'\\''c");
    }

    #[test]
    fn ssh_config_defaults() {
        let cfg = SshConfig::new("example.com", "deploy");
        assert_eq!(cfg.host(), "example.com");
        assert_eq!(cfg.port(), 22);
        assert_eq!(cfg.user(), "deploy");
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::Verify);
        assert_eq!(cfg.timeout(), std::time::Duration::from_secs(10));
        assert_eq!(
            cfg.keepalive_interval(),
            Some(std::time::Duration::from_secs(30))
        );
        assert!(cfg.socket().is_none());
    }

    #[test]
    fn ssh_config_builder() {
        let cfg = SshConfig::new("host", "user")
            .with_port(2222)
            .with_host_key_policy(HostKeyPolicy::Insecure)
            .with_timeout(std::time::Duration::from_secs(30))
            .with_keepalive(None)
            .with_socket(TmuxSocket::Name("test".into()))
            .unwrap();
        assert_eq!(cfg.port(), 2222);
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::Insecure);
        assert_eq!(cfg.timeout(), std::time::Duration::from_secs(30));
        assert_eq!(cfg.keepalive_interval(), None);
        assert_eq!(cfg.socket(), Some(&TmuxSocket::Name("test".into())));
    }

    #[test]
    fn ssh_config_is_localhost() {
        assert!(SshConfig::new("localhost", "u").is_localhost());
        assert!(SshConfig::new("127.0.0.1", "u").is_localhost());
        assert!(SshConfig::new("::1", "u").is_localhost());
        assert!(!SshConfig::new("remote", "u").is_localhost());
    }
}
