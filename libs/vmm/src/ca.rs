//! SSH Certificate Authority for guest authentication.
//!
//! Uses russh's re-exported ssh_key types to avoid version mismatches.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use russh::keys::ssh_key::certificate::{Builder, CertType};
use russh::keys::{Algorithm, Certificate, PrivateKey, PublicKey};
use thiserror::Error;

/// Default TTL for ephemeral guest certificates.
const DEFAULT_CERT_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Error)]
pub enum CaError {
    #[error("failed to generate CA keypair: {0}")]
    KeyGeneration(String),
    #[error("failed to sign certificate: {0}")]
    CertSigning(String),
}

pub struct SshCa {
    ca_key: PrivateKey,
    cert_ttl: Duration,
}

pub struct EphemeralCert {
    pub key: PrivateKey,
    pub cert: Certificate,
}

impl SshCa {
    pub fn new() -> Result<Self, CaError> {
        let ca_key = PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)
            .map_err(|e| CaError::KeyGeneration(e.to_string()))?;
        Ok(Self {
            ca_key,
            cert_ttl: DEFAULT_CERT_TTL,
        })
    }

    pub fn with_ttl(cert_ttl: Duration) -> Result<Self, CaError> {
        let mut ca = Self::new()?;
        ca.cert_ttl = cert_ttl;
        Ok(ca)
    }

    pub fn public_key(&self) -> &PublicKey {
        self.ca_key.public_key()
    }

    pub fn public_key_openssh(&self) -> Result<String, CaError> {
        self.ca_key
            .public_key()
            .to_openssh()
            .map_err(|e| CaError::CertSigning(e.to_string()))
    }

    pub fn sign_ephemeral(&self, guest_name: &str) -> Result<EphemeralCert, CaError> {
        let ephemeral_key = PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)
            .map_err(|e| CaError::KeyGeneration(e.to_string()))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| CaError::CertSigning(format!("system clock error: {e}")))?
            .as_secs();

        let mut builder = Builder::new_with_random_nonce(
            &mut rand::rng(),
            ephemeral_key.public_key().key_data().clone(),
            now,
            now + self.cert_ttl.as_secs(),
        )
        .map_err(|e| CaError::CertSigning(e.to_string()))?;

        builder
            .cert_type(CertType::User)
            .map_err(|e| CaError::CertSigning(e.to_string()))?;
        builder
            .valid_principal(guest_name)
            .map_err(|e| CaError::CertSigning(e.to_string()))?;
        for ext in [
            "permit-pty",
            "permit-user-rc",
            "permit-port-forwarding",
            "permit-agent-forwarding",
            "permit-X11-forwarding",
        ] {
            builder
                .extension(ext, "")
                .map_err(|e| CaError::CertSigning(e.to_string()))?;
        }

        let cert = builder
            .sign(&self.ca_key)
            .map_err(|e| CaError::CertSigning(e.to_string()))?;

        Ok(EphemeralCert {
            key: ephemeral_key,
            cert,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ca_generates_valid_keypair() {
        let ca = SshCa::new().unwrap();
        let pubkey_str = ca.public_key_openssh().unwrap();
        assert!(pubkey_str.starts_with("ssh-ed25519 "));
    }

    #[test]
    fn sign_ephemeral_produces_user_cert_with_principal() {
        let ca = SshCa::new().unwrap();
        let eph = ca.sign_ephemeral("alice").unwrap();
        assert_eq!(eph.cert.cert_type(), CertType::User);
        let principals: Vec<&str> = eph
            .cert
            .valid_principals()
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(principals.contains(&"alice"));
        assert_eq!(
            eph.cert.extensions().get("permit-pty").map(|s| s.as_str()),
            Some("")
        );
    }

    #[test]
    fn custom_ttl_is_respected() {
        let ca = SshCa::with_ttl(Duration::from_secs(30)).unwrap();
        assert_eq!(ca.cert_ttl, Duration::from_secs(30));
    }

    #[test]
    fn cert_validates_with_ssh_keygen() {
        let ca = SshCa::new().unwrap();
        let eph = ca.sign_ephemeral("alice").unwrap();

        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pub");
        std::fs::write(&cert_path, eph.cert.to_openssh().unwrap()).unwrap();

        let output = std::process::Command::new("ssh-keygen")
            .args(["-L", "-f"])
            .arg(&cert_path)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("=== ssh-keygen -L ===\n{stdout}");
        assert!(output.status.success());
        assert!(stdout.contains("alice"));
    }
}
