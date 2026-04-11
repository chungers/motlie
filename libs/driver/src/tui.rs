#[cfg(feature = "tui")]
pub struct TuiFrontend;

#[cfg(feature = "tui")]
impl TuiFrontend {
    pub fn new() -> Self {
        Self
    }
}
