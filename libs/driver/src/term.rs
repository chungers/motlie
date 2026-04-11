#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermBackendKind {
    Plain,
    #[cfg(feature = "term-vt100")]
    Vt100,
    Shadow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermSize {
    pub cols: u16,
    pub rows: u16,
}
