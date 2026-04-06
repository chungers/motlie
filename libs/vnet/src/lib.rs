//! motlie-vnet: Embeddable user-mode networking for virtual machines.
//!
//! Provides a vhost-user-net backend backed by libslirp for rootless guest
//! networking. Each guest gets its own backend instance with isolated network
//! state (DHCP, DNS, TCP/IP). No host capabilities or network namespaces required.
//!
//! # Architecture
//!
//! Two threads per guest:
//! - **Slirp thread** (!Send — owns the libslirp Context): runs the poll-based
//!   event loop, translates L2 frames to/from host L4 sockets.
//! - **Vhost-user worker thread** (Send+Sync — epoll-based): processes virtqueue
//!   kicks from Cloud Hypervisor, bridges frames to/from the slirp thread via channels.
//!
//! This design avoids unsafe Send impls on libslirp's C state.

pub mod slirp;
mod backend;
mod error;
mod config;

pub use config::{PortForward, VnetConfig};
pub use error::VnetError;
pub use backend::{VnetBackend, VnetHandle};
