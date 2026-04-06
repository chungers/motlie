//! SSH Certificate Authority for guest authentication.
//!
//! The CA holds an Ed25519 keypair in memory and signs ephemeral certificates
//! for the SSH proxy to authenticate to guest VMs. Each cert has a short TTL
//! (default 60s) and a principal matching the guest identity.
//!
//! ## Design (FR-6, DESIGN.md)
//!
//! - Outbound auth: CA signs throwaway Ed25519 certs with `principal=<guest>`
//! - Guest sshd validates: cert signed by CA? principal in auth_principals? not expired?
//! - No user-side key management — the orchestrator acts as both CA and SSH client

use std::time::Duration;

use ssh_key::{
    certificate::{Builder, CertType},
    PrivateKey, PublicKey,
};
use thiserror::Error;

/// Default TTL for ephemeral guest certificates.
const DEFAULT_CERT_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Error)]
pub enum CaError {
    #[error("failed to generate CA keypair: {0}")]
    KeyGeneration(String),
    #[error("failed to sign certificate: {0}")]
    CertSigning(String),
}

/// In-memory SSH Certificate Authority.
///
/// Generates an Ed25519 CA keypair on construction. Signs ephemeral
/// user certificates for the SSH proxy to authenticate to guest sshd.
pub struct SshCa {
    /// The CA's signing keypair (private + public).
    ca_key: PrivateKey,
    /// Certificate TTL.
    cert_ttl: Duration,
}

/// An ephemeral keypair + certificate for a single guest connection.
pub struct EphemeralCert {
    /// The throwaway private key (used by russh client to authenticate).
    pub key: PrivateKey,
    /// The signed certificate (presented to guest sshd).
    pub cert: ssh_key::Certificate,
}

impl SshCa {
    /// Create a new CA with a freshly generated Ed25519 keypair.
    pub fn new() -> Result<Self, CaError> {
        let ca_key = PrivateKey::random(&mut rand::thread_rng(), ssh_key::Algorithm::Ed25519)
            .map_err(|e| CaError::KeyGeneration(e.to_string()))?;

        Ok(Self {
            ca_key,
            cert_ttl: DEFAULT_CERT_TTL,
        })
    }

    /// Create a CA with a custom certificate TTL.
    pub fn with_ttl(cert_ttl: Duration) -> Result<Self, CaError> {
        let mut ca = Self::new()?;
        ca.cert_ttl = cert_ttl;
        Ok(ca)
    }

    /// The CA's public key, for baking into guest images at
    /// `/etc/ssh/ca/user_ca.pub`.
    pub fn public_key(&self) -> &PublicKey {
        self.ca_key.public_key()
    }

    /// The CA's public key in OpenSSH format (for writing to files).
    pub fn public_key_openssh(&self) -> Result<String, CaError> {
        self.ca_key
            .public_key()
            .to_openssh()
            .map_err(|e| CaError::CertSigning(e.to_string()))
    }

    /// Sign an ephemeral certificate for connecting to a guest VM.
    ///
    /// Generates a fresh Ed25519 keypair and signs a user certificate with:
    /// - `principal` = guest name (e.g. "alice")
    /// - `valid_after` = now
    /// - `valid_before` = now + cert_ttl
    /// - `cert_type` = User
    ///
    /// The returned `EphemeralCert` is used once by the SSH proxy to
    /// authenticate to the guest's sshd, then discarded.
    pub fn sign_ephemeral(&self, guest_name: &str) -> Result<EphemeralCert, CaError> {
        let ephemeral_key =
            PrivateKey::random(&mut rand::thread_rng(), ssh_key::Algorithm::Ed25519)
                .map_err(|e| CaError::KeyGeneration(e.to_string()))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut builder = Builder::new_with_random_nonce(
            &mut rand::thread_rng(),
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

        // Cert should be a user cert
        assert_eq!(eph.cert.cert_type(), CertType::User);

        // Principal should contain the guest name
        let principals: Vec<&str> = eph.cert.valid_principals().iter().map(|s| s.as_str()).collect();
        assert!(principals.contains(&"alice"), "expected 'alice' in principals: {:?}", principals);

        // Ephemeral key should differ from CA key
        assert_ne!(
            eph.key.public_key().to_openssh().unwrap(),
            ca.public_key_openssh().unwrap()
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

        // Write cert to temp file and validate with ssh-keygen -L
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pub");
        let ca_path = dir.path().join("ca.pub");

        std::fs::write(&cert_path, eph.cert.to_openssh().unwrap()).unwrap();
        std::fs::write(&ca_path, ca.public_key_openssh().unwrap()).unwrap();

        // ssh-keygen -L shows cert details
        let output = std::process::Command::new("ssh-keygen")
            .args(["-L", "-f"])
            .arg(&cert_path)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("=== ssh-keygen -L ===\n{stdout}");
        assert!(output.status.success(), "ssh-keygen -L failed: {}", String::from_utf8_lossy(&output.stderr));
        assert!(stdout.contains("alice"), "cert should contain principal 'alice': {stdout}");
        assert!(stdout.contains("Type: ssh-ed25519-cert-v01@openssh.com user"), "should be user cert: {stdout}");
    }
}
