//! Authentication middleware for MCP server
//!
//! Provides token-based authentication with extensibility for future auth methods.

use anyhow::Result;

/// Authentication context for MCP requests
#[derive(Clone, Debug)]
pub struct AuthContext {
    /// The expected authentication token (None means no auth required)
    expected_token: Option<String>,
}

impl AuthContext {
    /// Create a new authentication context with optional token
    pub fn new(token: Option<String>) -> Self {
        Self {
            expected_token: token,
        }
    }

    /// Authenticate a request with the provided token
    ///
    /// # Arguments
    /// * `provided_token` - The token provided in the request
    ///
    /// # Returns
    /// * `Ok(())` if authentication succeeds
    /// * `Err(_)` if authentication fails
    pub fn authenticate(&self, provided_token: Option<&str>) -> Result<()> {
        match (&self.expected_token, provided_token) {
            // Auth required and correct token provided
            (Some(expected), Some(provided)) if expected == provided => {
                log::debug!("Authentication successful");
                Ok(())
            }
            // No auth required
            (None, _) => {
                log::debug!("No authentication required");
                Ok(())
            }
            // Auth required but wrong/missing token
            (Some(_), None) => {
                log::warn!("Authentication failed: missing token");
                Err(anyhow::anyhow!("Authentication token required"))
            }
            (Some(_), Some(_)) => {
                log::warn!("Authentication failed: invalid token");
                Err(anyhow::anyhow!("Invalid authentication token"))
            }
        }
    }

    /// Check if authentication is enabled
    pub fn is_enabled(&self) -> bool {
        self.expected_token.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_no_token_required() {
        let auth = AuthContext::new(None);
        assert!(!auth.is_enabled());
        assert!(auth.authenticate(None).is_ok());
        assert!(auth.authenticate(Some("any-token")).is_ok());
    }

    #[test]
    fn test_auth_with_correct_token() {
        let auth = AuthContext::new(Some("secret123".to_string()));
        assert!(auth.is_enabled());
        assert!(auth.authenticate(Some("secret123")).is_ok());
    }

    #[test]
    fn test_auth_with_wrong_token() {
        let auth = AuthContext::new(Some("secret123".to_string()));
        let result = auth.authenticate(Some("wrong-token"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Invalid authentication token");
    }

    #[test]
    fn test_auth_with_missing_token() {
        let auth = AuthContext::new(Some("secret123".to_string()));
        let result = auth.authenticate(None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Authentication token required");
    }
}
