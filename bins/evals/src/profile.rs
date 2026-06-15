//! Closed registry of eval **Profiles** — the runtime layer of the #521 coverage
//! ontology. A Profile is the `(arch, accelerator)` instantiation the matrix runs
//! on; it is the third axis of the coverage tuple `(CuratedBundle,
//! QuantizationScheme, CapabilityKind, Profile)`.
//!
//! Promoting the five profile names from bare strings (which drifted through
//! `snapshot.rs`/`accelerator.rs`) into a closed enum gives one canonical label
//! set, identical in code and in eval data, and the typed `Profile ->
//! Accelerator` bridge (`accelerator()`) that joins the runtime profile to the
//! model-side `motlie_model::Accelerator` the compile-time declaration is keyed
//! on (`BackendKind::accel_support`).

use motlie_model::Accelerator;

use crate::result::AcceleratorClass;

/// Host architecture component of a Profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
}

/// The closed set of eval profiles. The canonical id strings are the single
/// source of truth, matching the existing on-disk profile labels exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Profile {
    LocalCpuX86_64,
    LocalCpuAarch64,
    AppleMetal,
    DgxSpark,
    CudaWorkstation,
}

impl Profile {
    /// Every profile, for exhaustive coverage walks (the Profile axis of the
    /// completeness matrix).
    pub const ALL: [Profile; 5] = [
        Self::LocalCpuX86_64,
        Self::LocalCpuAarch64,
        Self::AppleMetal,
        Self::DgxSpark,
        Self::CudaWorkstation,
    ];

    /// Canonical profile id strings, in `ALL` order. Mirrors the
    /// `CuratedBundle::CANONICAL_IDS` pattern (#518) for the Profile axis.
    pub const CANONICAL_IDS: [&'static str; 5] = [
        "local-cpu-x86_64",
        "local-cpu-aarch64",
        "apple-metal",
        "dgx-spark",
        "cuda-workstation",
    ];

    pub fn canonical_id(self) -> &'static str {
        match self {
            Self::LocalCpuX86_64 => "local-cpu-x86_64",
            Self::LocalCpuAarch64 => "local-cpu-aarch64",
            Self::AppleMetal => "apple-metal",
            Self::DgxSpark => "dgx-spark",
            Self::CudaWorkstation => "cuda-workstation",
        }
    }

    /// Parse a profile id string into the closed enum at the data/CLI boundary.
    /// Returns `None` for any non-canonical string (no silent substring guess).
    pub fn from_id(id: &str) -> Option<Profile> {
        match id {
            "local-cpu-x86_64" => Some(Self::LocalCpuX86_64),
            "local-cpu-aarch64" => Some(Self::LocalCpuAarch64),
            "apple-metal" => Some(Self::AppleMetal),
            "dgx-spark" => Some(Self::DgxSpark),
            "cuda-workstation" => Some(Self::CudaWorkstation),
            _ => None,
        }
    }

    pub fn arch(self) -> Arch {
        match self {
            Self::LocalCpuX86_64 | Self::CudaWorkstation => Arch::X86_64,
            Self::LocalCpuAarch64 | Self::AppleMetal | Self::DgxSpark => Arch::Aarch64,
        }
    }

    /// The model-side accelerator this profile targets — the typed bridge that
    /// joins a runtime Profile to `BackendKind::accel_support`. Total and exact:
    /// every profile maps to exactly one physical `Accelerator` (no `Any`).
    pub fn accelerator(self) -> Accelerator {
        match self {
            Self::LocalCpuX86_64 | Self::LocalCpuAarch64 => Accelerator::Cpu,
            Self::AppleMetal => Accelerator::Metal,
            Self::DgxSpark | Self::CudaWorkstation => Accelerator::Cuda,
        }
    }

    /// The eval-runtime `AcceleratorClass` (carries `Any`/`Unavailable` for
    /// resolution states) corresponding to this profile's target accelerator.
    pub fn accelerator_class(self) -> AcceleratorClass {
        match self.accelerator() {
            Accelerator::Cpu => AcceleratorClass::Cpu,
            Accelerator::Cuda => AcceleratorClass::Cuda,
            Accelerator::Metal => AcceleratorClass::Metal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_ids_round_trip_and_match_all() {
        // Every variant round-trips through its canonical id, and ALL lines up
        // 1-1 with CANONICAL_IDS — the closed-set completeness guarantee.
        assert_eq!(Profile::ALL.len(), Profile::CANONICAL_IDS.len());
        for (profile, id) in Profile::ALL.iter().zip(Profile::CANONICAL_IDS.iter()) {
            assert_eq!(profile.canonical_id(), *id);
            assert_eq!(Profile::from_id(id), Some(*profile));
        }
    }

    #[test]
    fn from_id_rejects_non_canonical() {
        assert_eq!(Profile::from_id("dgx"), None);
        assert_eq!(Profile::from_id("cuda"), None);
        assert_eq!(Profile::from_id(""), None);
    }

    #[test]
    fn accelerator_bridge_is_exact() {
        assert_eq!(Profile::LocalCpuX86_64.accelerator(), Accelerator::Cpu);
        assert_eq!(Profile::LocalCpuAarch64.accelerator(), Accelerator::Cpu);
        assert_eq!(Profile::AppleMetal.accelerator(), Accelerator::Metal);
        assert_eq!(Profile::DgxSpark.accelerator(), Accelerator::Cuda);
        assert_eq!(Profile::CudaWorkstation.accelerator(), Accelerator::Cuda);
    }

    #[test]
    fn arch_split_matches_profiles() {
        assert_eq!(Profile::LocalCpuX86_64.arch(), Arch::X86_64);
        assert_eq!(Profile::CudaWorkstation.arch(), Arch::X86_64);
        assert_eq!(Profile::DgxSpark.arch(), Arch::Aarch64);
        assert_eq!(Profile::AppleMetal.arch(), Arch::Aarch64);
    }
}
