use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlatformSnapshot {
    pub os: Option<String>,
    pub arch: Option<String>,
    pub target_triple: Option<String>,
    pub hostname: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
    pub total_swap_bytes: Option<u64>,
    pub free_swap_bytes: Option<u64>,
    pub gpu_backend: Option<String>,
    pub gpus: Vec<GpuSnapshot>,
    pub unavailable: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub backend: Option<String>,
    pub model: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub free_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlatformCollector {
    profile_name: String,
}

impl PlatformCollector {
    pub fn new(profile_name: impl Into<String>) -> Self {
        Self {
            profile_name: profile_name.into(),
        }
    }

    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    pub fn collect(&self) -> PlatformSnapshot {
        let mut system = System::new();
        system.refresh_memory();

        PlatformSnapshot {
            os: Some(std::env::consts::OS.to_owned()),
            arch: Some(std::env::consts::ARCH.to_owned()),
            target_triple: option_env!("TARGET").map(str::to_owned),
            hostname: std::env::var("HOSTNAME").ok(),
            total_memory_bytes: Some(system.total_memory()),
            available_memory_bytes: Some(system.available_memory()),
            total_swap_bytes: Some(system.total_swap()),
            free_swap_bytes: Some(system.free_swap()),
            gpu_backend: Some("unavailable".to_owned()),
            gpus: Vec::new(),
            unavailable: vec![
                "kernel_version".to_owned(),
                "libc".to_owned(),
                "gpu_inventory".to_owned(),
                format!("profile:{}:accelerator_metadata", self.profile_name),
            ],
        }
    }
}
