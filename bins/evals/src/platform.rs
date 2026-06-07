use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlatformProfile {
    pub name: String,
    pub os: Option<String>,
    pub arch: Option<String>,
    pub target_triple: Option<String>,
}

impl PlatformProfile {
    pub fn current(profile_name: impl Into<String>) -> Self {
        Self {
            name: profile_name.into(),
            os: Some(std::env::consts::OS.to_owned()),
            arch: Some(std::env::consts::ARCH.to_owned()),
            target_triple: option_env!("TARGET").map(str::to_owned),
        }
    }
}
