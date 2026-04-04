/// Errors from the vnet backend lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum VnetError {
    /// Socket path validation failed (not writable, parent missing).
    #[error("invalid socket path: {0}")]
    SocketPath(String),

    /// Socket bind failed (EADDRINUSE, EACCES).
    #[error("socket bind failed: {0}")]
    SocketBind(#[source] std::io::Error),

    /// Stale socket cleanup failed before bind.
    #[error("socket cleanup failed: {0}")]
    SocketCleanup(#[source] std::io::Error),

    /// libslirp context creation failed.
    #[error("slirp init failed: {0}")]
    SlirpInit(String),

    /// vhost-user backend initialization failed.
    #[error("backend init failed: {0}")]
    BackendInit(String),

    /// Host DNS resolver discovery/parsing failed.
    #[error("DNS resolver error: {0}")]
    DnsResolver(String),

    /// Port forward bind failed (EADDRINUSE on host-side listener).
    #[error("port forward bind failed on host port {host_port}: {source}")]
    PortForwardBind {
        host_port: u16,
        #[source]
        source: std::io::Error,
    },
}
