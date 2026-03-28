//! simple_host: proof-of-concept host server with rustyline REPL.
//!
//! Runs an FsServer with MemOverlay, serves over vsock, and provides
//! an interactive REPL for overlay mutation (put, whiteout, rm, ls, etc.).
//!
//! Usage: cargo run -p motlie-vfs --example simple_host

fn main() {
    // Placeholder — Phase 5.1 implements the full host REPL harness.
    eprintln!("simple_host: not yet implemented");
    std::process::exit(1);
}
