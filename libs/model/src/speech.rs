//! Shared TTS data types.
//!
//! The typed synthesis contracts live in [`crate::typed`]. This module only
//! carries synthesis parameter controls that are shared across bundles and
//! backends.

/// Stream-scoped speech synthesis parameters.
///
/// Backends may support only a subset of these controls. The v1 Piper backend
/// supports `speaking_rate` and rejects `seed` with `InvalidConfiguration`.
/// Future backends may accept a larger subset without changing the public API.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechParams {
    pub speaking_rate: Option<f32>,
    pub seed: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_params_defaults_to_no_optional_controls() {
        let params = SpeechParams::default();

        assert_eq!(params.speaking_rate, None);
        assert_eq!(params.seed, None);
    }
}
