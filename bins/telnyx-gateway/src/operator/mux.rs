#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandSourceKind {
    Tui,
    Socket,
    Replay,
}
