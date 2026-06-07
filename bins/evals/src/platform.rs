use std::collections::BTreeMap;
use std::process::Command;

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
    pub accelerator_metadata: BTreeMap<String, String>,
    pub unavailable: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub index: Option<u32>,
    pub backend: Option<String>,
    pub id: Option<String>,
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
        let nvidia = collect_nvidia_smi();

        let mut unavailable = vec!["kernel_version".to_owned(), "libc".to_owned()];
        let (gpu_backend, gpus, accelerator_metadata) = match nvidia {
            Some(inventory) => (
                Some("nvidia".to_owned()),
                inventory.gpus,
                inventory.accelerator_metadata,
            ),
            None => {
                unavailable.push("gpu_inventory".to_owned());
                unavailable.push(format!(
                    "profile:{}:accelerator_metadata",
                    self.profile_name
                ));
                (Some("unavailable".to_owned()), Vec::new(), BTreeMap::new())
            }
        };

        PlatformSnapshot {
            os: Some(std::env::consts::OS.to_owned()),
            arch: Some(std::env::consts::ARCH.to_owned()),
            target_triple: option_env!("TARGET").map(str::to_owned),
            hostname: std::env::var("HOSTNAME").ok(),
            total_memory_bytes: Some(system.total_memory()),
            available_memory_bytes: Some(system.available_memory()),
            total_swap_bytes: Some(system.total_swap()),
            free_swap_bytes: Some(system.free_swap()),
            gpu_backend,
            gpus,
            accelerator_metadata,
            unavailable,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NvidiaInventory {
    gpus: Vec<GpuSnapshot>,
    accelerator_metadata: BTreeMap<String, String>,
}

fn collect_nvidia_smi() -> Option<NvidiaInventory> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let (gpus, driver_version) = parse_nvidia_gpu_query(&stdout);
    if gpus.is_empty() {
        return None;
    }

    let mut accelerator_metadata = BTreeMap::new();
    accelerator_metadata.insert("collector".to_owned(), "nvidia-smi".to_owned());
    if let Some(driver_version) = driver_version {
        accelerator_metadata.insert("driver_version".to_owned(), driver_version);
    }
    if let Some(cuda_version) = query_nvidia_cuda_version() {
        accelerator_metadata.insert("cuda_version".to_owned(), cuda_version);
    }

    Some(NvidiaInventory {
        gpus,
        accelerator_metadata,
    })
}

fn query_nvidia_cuda_version() -> Option<String> {
    let output = Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_nvidia_cuda_version(&stdout)
}

fn parse_nvidia_gpu_query(raw: &str) -> (Vec<GpuSnapshot>, Option<String>) {
    let mut gpus = Vec::new();
    let mut driver_version = None;

    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 6 {
            continue;
        }

        if driver_version.is_none() {
            driver_version = parse_optional_string(fields[5]);
        }

        gpus.push(GpuSnapshot {
            index: fields[0].parse::<u32>().ok(),
            backend: Some("nvidia".to_owned()),
            id: parse_optional_string(fields[2]),
            model: parse_optional_string(fields[1]),
            total_memory_bytes: parse_nvidia_memory_mib(fields[3]),
            free_memory_bytes: parse_nvidia_memory_mib(fields[4]),
        });
    }

    (gpus, driver_version)
}

fn parse_nvidia_cuda_version(raw: &str) -> Option<String> {
    let marker = "CUDA Version:";
    let (_, rest) = raw.split_once(marker)?;
    rest.split_whitespace().next().map(str::to_owned)
}

fn parse_nvidia_memory_mib(raw: &str) -> Option<u64> {
    let value = parse_optional_string(raw)?;
    let mib = value.parse::<u64>().ok()?;
    Some(mib.saturating_mul(1024 * 1024))
}

fn parse_optional_string(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "N/A" || trimmed == "[N/A]" {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_records_nvidia_identity_when_nvidia_smi_is_available() {
        if collect_nvidia_smi().is_none() {
            return;
        }

        let snapshot = PlatformCollector::new("dgx-spark").collect();

        assert_eq!(snapshot.gpu_backend.as_deref(), Some("nvidia"));
        assert!(!snapshot.gpus.is_empty());
        assert_eq!(snapshot.gpus[0].backend.as_deref(), Some("nvidia"));
        assert!(!snapshot.accelerator_metadata.is_empty());
        assert!(!snapshot
            .unavailable
            .iter()
            .any(|field| field == "gpu_inventory"));
    }

    #[test]
    fn parses_nvidia_gpu_query_with_unavailable_memory() {
        let (gpus, driver_version) =
            parse_nvidia_gpu_query("0, NVIDIA GB10, GPU-test-id, [N/A], [N/A], 580.159.03\n");

        assert_eq!(driver_version.as_deref(), Some("580.159.03"));
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].index, Some(0));
        assert_eq!(gpus[0].backend.as_deref(), Some("nvidia"));
        assert_eq!(gpus[0].id.as_deref(), Some("GPU-test-id"));
        assert_eq!(gpus[0].model.as_deref(), Some("NVIDIA GB10"));
        assert_eq!(gpus[0].total_memory_bytes, None);
        assert_eq!(gpus[0].free_memory_bytes, None);
    }

    #[test]
    fn parses_nvidia_gpu_query_memory_as_mib() {
        let (gpus, driver_version) =
            parse_nvidia_gpu_query("0, NVIDIA RTX, GPU-test-id, 24576, 12288, 580.159.03\n");

        assert_eq!(driver_version.as_deref(), Some("580.159.03"));
        assert_eq!(gpus[0].total_memory_bytes, Some(24_576 * 1024 * 1024));
        assert_eq!(gpus[0].free_memory_bytes, Some(12_288 * 1024 * 1024));
    }

    #[test]
    fn parses_cuda_version_from_nvidia_smi_banner() {
        let raw = "| NVIDIA-SMI 580.159.03  Driver Version: 580.159.03  CUDA Version: 13.0 |";

        assert_eq!(parse_nvidia_cuda_version(raw).as_deref(), Some("13.0"));
    }
}
