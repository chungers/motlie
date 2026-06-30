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
    #[serde(default)]
    pub host_id: Option<String>,
    #[serde(default)]
    pub host_slug: Option<String>,
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
    #[serde(default)]
    pub unified_memory: Option<bool>,
    #[serde(default)]
    pub recommended_max_working_set_size_bytes: Option<u64>,
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
        let metal = collect_metal();

        let mut unavailable = vec!["kernel_version".to_owned(), "libc".to_owned()];
        let (gpu_backend, gpus, accelerator_metadata) = match (nvidia, metal) {
            (Some(inventory), _) => (
                Some("nvidia".to_owned()),
                inventory.gpus,
                inventory.accelerator_metadata,
            ),
            (None, Some(inventory)) => (
                Some("metal".to_owned()),
                inventory.gpus,
                inventory.accelerator_metadata,
            ),
            (None, None) => {
                unavailable.push("gpu_inventory".to_owned());
                unavailable.push(format!(
                    "profile:{}:accelerator_metadata",
                    self.profile_name
                ));
                (Some("unavailable".to_owned()), Vec::new(), BTreeMap::new())
            }
        };
        if gpu_backend.as_deref() == Some("metal") {
            unavailable.push("metal_usability_probe".to_owned());
            unavailable.push("metal_recommended_max_working_set_size_bytes".to_owned());
        }

        let hostname = portable_hostname();
        let host_slug = hostname.as_deref().map(sanitize_slug);

        PlatformSnapshot {
            os: Some(std::env::consts::OS.to_owned()),
            arch: Some(std::env::consts::ARCH.to_owned()),
            target_triple: option_env!("TARGET").map(str::to_owned),
            hostname: hostname.clone(),
            host_id: hostname.clone(),
            host_slug,
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
struct AcceleratorInventory {
    gpus: Vec<GpuSnapshot>,
    accelerator_metadata: BTreeMap<String, String>,
}

fn collect_nvidia_smi() -> Option<AcceleratorInventory> {
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

    Some(AcceleratorInventory {
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

fn collect_metal() -> Option<AcceleratorInventory> {
    #[cfg(target_os = "macos")]
    {
        collect_metal_system_profiler()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn collect_metal_system_profiler() -> Option<AcceleratorInventory> {
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_metal_system_profiler(&stdout)
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
            unified_memory: Some(false),
            recommended_max_working_set_size_bytes: None,
        });
    }

    (gpus, driver_version)
}

#[cfg(any(target_os = "macos", test))]
fn parse_metal_system_profiler(raw: &str) -> Option<AcceleratorInventory> {
    if !raw.to_ascii_lowercase().contains("metal") {
        return None;
    }

    let model = raw.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("Chipset Model:")
            .or_else(|| trimmed.strip_prefix("Model:"))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    });
    let id = raw.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("Device ID:")
            .or_else(|| trimmed.strip_prefix("Vendor ID:"))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    });

    let mut accelerator_metadata = BTreeMap::new();
    accelerator_metadata.insert("collector".to_owned(), "system_profiler".to_owned());
    accelerator_metadata.insert(
        "metal_probe".to_owned(),
        "system_profiler_presence_only".to_owned(),
    );
    accelerator_metadata.insert(
        "metal_usability_probe".to_owned(),
        "not_instrumented".to_owned(),
    );
    accelerator_metadata.insert("unified_memory".to_owned(), "true".to_owned());
    accelerator_metadata.insert(
        "recommended_max_working_set_size_bytes".to_owned(),
        "unavailable".to_owned(),
    );

    Some(AcceleratorInventory {
        gpus: vec![GpuSnapshot {
            index: Some(0),
            backend: Some("metal".to_owned()),
            id,
            model: model.or_else(|| Some("Apple Metal device".to_owned())),
            total_memory_bytes: None,
            free_memory_bytes: None,
            unified_memory: Some(true),
            recommended_max_working_set_size_bytes: None,
        }],
        accelerator_metadata,
    })
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

pub fn portable_hostname() -> Option<String> {
    System::host_name()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| command_output("hostname", &[]))
        .or_else(|| command_output("uname", &["-n"]))
        .or_else(|| command_output("scutil", &["--get", "LocalHostName"]))
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

pub fn sanitize_slug(value: &str) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            previous_dash = false;
        } else if !previous_dash {
            slug.push('-');
            previous_dash = true;
        }
    }
    let slug = slug.trim_matches('-').to_owned();
    if slug.is_empty() {
        "unknown".to_owned()
    } else {
        slug
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
        assert_eq!(gpus[0].unified_memory, Some(false));
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

    #[test]
    fn parses_metal_system_profiler_inventory() {
        let raw = "Graphics/Displays:\n\n    Apple M3 Max:\n\n      Chipset Model: Apple M3 Max\n      Metal Support: Metal 3\n";
        let inventory = parse_metal_system_profiler(raw).unwrap();

        assert_eq!(inventory.gpus[0].backend.as_deref(), Some("metal"));
        assert_eq!(inventory.gpus[0].model.as_deref(), Some("Apple M3 Max"));
        assert_eq!(inventory.gpus[0].unified_memory, Some(true));
        assert_eq!(
            inventory
                .accelerator_metadata
                .get("metal_probe")
                .map(String::as_str),
            Some("system_profiler_presence_only")
        );
        assert_eq!(
            inventory
                .accelerator_metadata
                .get("recommended_max_working_set_size_bytes")
                .map(String::as_str),
            Some("unavailable")
        );
    }

    #[test]
    fn slug_sanitizes_hostnames() {
        assert_eq!(sanitize_slug("MacBook Pro.local"), "macbook-pro-local");
        assert_eq!(sanitize_slug("***"), "unknown");
    }
}
