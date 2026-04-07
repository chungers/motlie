use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdminNetMode {
    None,
    Tap,
}

impl AdminNetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tap => "tap",
        }
    }
}

impl FromStr for AdminNetMode {
    type Err = NetworkModeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "tap" => Ok(Self::Tap),
            _ => Err(NetworkModeParseError::InvalidAdminMode(value.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressNetMode {
    None,
    Tap,
    VhostUser,
}

impl EgressNetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tap => "tap",
            Self::VhostUser => "vhost-user",
        }
    }
}

impl FromStr for EgressNetMode {
    type Err = NetworkModeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "tap" => Ok(Self::Tap),
            "vhost-user" => Ok(Self::VhostUser),
            _ => Err(NetworkModeParseError::InvalidEgressMode(value.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkModes {
    pub admin: AdminNetMode,
    pub egress: EgressNetMode,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NetworkModeParseError {
    #[error("admin-net must be one of: none, tap (got {0})")]
    InvalidAdminMode(String),
    #[error("egress-net must be one of: none, tap, vhost-user (got {0})")]
    InvalidEgressMode(String),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NetworkModeError {
    #[error(
        "supported launch modes are --admin-net=none --egress-net=none, --admin-net=none --egress-net=vhost-user, --admin-net=tap --egress-net=tap, and --admin-net=tap --egress-net=vhost-user"
    )]
    UnsupportedCombination {
        admin: AdminNetMode,
        egress: EgressNetMode,
    },
}

pub fn validate_network_modes(modes: NetworkModes) -> Result<(), NetworkModeError> {
    match (modes.admin, modes.egress) {
        (AdminNetMode::None, EgressNetMode::None)
        | (AdminNetMode::None, EgressNetMode::VhostUser)
        | (AdminNetMode::Tap, EgressNetMode::Tap)
        | (AdminNetMode::Tap, EgressNetMode::VhostUser) => Ok(()),
        _ => Err(NetworkModeError::UnsupportedCombination {
            admin: modes.admin,
            egress: modes.egress,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_network_modes() {
        assert_eq!("none".parse::<AdminNetMode>().unwrap(), AdminNetMode::None);
        assert_eq!("tap".parse::<AdminNetMode>().unwrap(), AdminNetMode::Tap);
        assert_eq!("none".parse::<EgressNetMode>().unwrap(), EgressNetMode::None);
        assert_eq!("tap".parse::<EgressNetMode>().unwrap(), EgressNetMode::Tap);
        assert_eq!(
            "vhost-user".parse::<EgressNetMode>().unwrap(),
            EgressNetMode::VhostUser
        );
    }

    #[test]
    fn reject_invalid_network_modes() {
        assert_eq!(
            "bridge".parse::<AdminNetMode>().unwrap_err(),
            NetworkModeParseError::InvalidAdminMode("bridge".to_string())
        );
        assert_eq!(
            "slirp".parse::<EgressNetMode>().unwrap_err(),
            NetworkModeParseError::InvalidEgressMode("slirp".to_string())
        );
    }

    #[test]
    fn validate_supported_combinations() {
        assert!(validate_network_modes(NetworkModes {
            admin: AdminNetMode::None,
            egress: EgressNetMode::None,
        })
        .is_ok());
        assert!(validate_network_modes(NetworkModes {
            admin: AdminNetMode::None,
            egress: EgressNetMode::VhostUser,
        })
        .is_ok());
        assert!(validate_network_modes(NetworkModes {
            admin: AdminNetMode::Tap,
            egress: EgressNetMode::Tap,
        })
        .is_ok());
        assert!(validate_network_modes(NetworkModes {
            admin: AdminNetMode::Tap,
            egress: EgressNetMode::VhostUser,
        })
        .is_ok());
    }

    #[test]
    fn reject_unsupported_combinations() {
        let err = validate_network_modes(NetworkModes {
            admin: AdminNetMode::None,
            egress: EgressNetMode::Tap,
        })
        .unwrap_err();
        assert_eq!(
            err,
            NetworkModeError::UnsupportedCombination {
                admin: AdminNetMode::None,
                egress: EgressNetMode::Tap,
            }
        );
    }
}
