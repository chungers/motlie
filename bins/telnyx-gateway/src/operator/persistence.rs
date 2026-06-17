use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::operator::config_file::render_state_toml;
use crate::operator::script::expand_user_path;
use crate::operator::state::GatewayState;

pub fn render_state_dump(state: &GatewayState) -> String {
    render_state_toml(state)
}

pub fn write_state_dump(path: &Path, state: &GatewayState) -> anyhow::Result<PathBuf> {
    let path = expand_user_path(path);
    let dump = render_state_dump(state);
    fs::write(&path, dump).with_context(|| format!("write state dump '{}'", path.display()))?;
    Ok(path)
}
