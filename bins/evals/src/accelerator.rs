use std::collections::BTreeMap;

use crate::platform::PlatformSnapshot;
use crate::result::{
    AcceleratorClass, AcceleratorDevice, AcceleratorSection, AcceptanceStatus, OutcomeReason,
};

pub fn requested_for_profile(profile: &str) -> AcceleratorClass {
    let profile = profile.to_ascii_lowercase();
    if profile.contains("cuda") || profile.contains("dgx") {
        AcceleratorClass::Cuda
    } else if profile.contains("metal") || profile.contains("apple") {
        AcceleratorClass::Metal
    } else if profile.contains("cpu") {
        AcceleratorClass::Cpu
    } else {
        AcceleratorClass::Any
    }
}

pub fn default_profile_for_platform(platform: &PlatformSnapshot) -> String {
    match platform.gpu_backend.as_deref() {
        Some("nvidia") => {
            if platform.gpus.iter().any(|gpu| {
                gpu.model
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .contains("gb10")
            }) {
                "dgx-spark".to_owned()
            } else {
                "cuda-workstation".to_owned()
            }
        }
        Some("metal") => "apple-metal".to_owned(),
        _ => match std::env::consts::ARCH {
            "aarch64" => "local-cpu-aarch64".to_owned(),
            "x86_64" => "local-cpu-x86_64".to_owned(),
            other => format!("local-cpu-{other}"),
        },
    }
}

pub fn resolve_for_profile(profile: &str, platform: &PlatformSnapshot) -> AcceleratorSection {
    resolve(requested_for_profile(profile), platform, None, None)
}

pub fn resolve(
    requested: AcceleratorClass,
    platform: &PlatformSnapshot,
    backend_mode: Option<String>,
    offload: Option<String>,
) -> AcceleratorSection {
    let available = available_class(platform);
    let caller_reported_backend = backend_mode.is_some() || offload.is_some();
    let mut backend_mode = backend_mode;
    let mut offload = offload.or_else(runtime_offload_override);
    let runtime_forced_cpu = runtime_forces_cpu() || runtime_gpu_layers_override() == Some(0);
    let resolved = if runtime_forced_cpu && requested != AcceleratorClass::Unavailable {
        AcceleratorClass::Cpu
    } else {
        match requested {
            AcceleratorClass::Any => {
                if available == AcceleratorClass::Unavailable {
                    AcceleratorClass::Cpu
                } else {
                    available
                }
            }
            AcceleratorClass::Cpu => AcceleratorClass::Cpu,
            AcceleratorClass::Cuda if available == AcceleratorClass::Cuda => AcceleratorClass::Cuda,
            AcceleratorClass::Metal if available == AcceleratorClass::Metal => {
                AcceleratorClass::Metal
            }
            AcceleratorClass::Cuda | AcceleratorClass::Metal | AcceleratorClass::Unavailable => {
                AcceleratorClass::Unavailable
            }
        }
    };

    if runtime_forced_cpu {
        backend_mode = Some("cpu".to_owned());
        if offload.is_none() {
            offload = Some("gpu_layers=0".to_owned());
        }
    }

    let mut fallback_reason = if requested == AcceleratorClass::Any
        || requested == resolved
        || (requested == AcceleratorClass::Cpu && resolved == AcceleratorClass::Cpu)
    {
        None
    } else if available == AcceleratorClass::Unavailable {
        Some(OutcomeReason::AcceleratorUnavailable)
    } else {
        Some(OutcomeReason::AcceleratorMismatch)
    };

    if matches!(requested, AcceleratorClass::Cuda | AcceleratorClass::Metal)
        && resolved == requested
        && !caller_reported_backend
    {
        fallback_reason = Some(OutcomeReason::BackendOffloadUnverified);
        backend_mode = Some("backend_offload_unverified".to_owned());
    }

    if backend_mode.is_none() {
        backend_mode = Some(match resolved {
            AcceleratorClass::Cpu => "cpu".to_owned(),
            AcceleratorClass::Unavailable => "unavailable".to_owned(),
            AcceleratorClass::Cuda | AcceleratorClass::Metal | AcceleratorClass::Any => {
                resolved.as_str().to_owned()
            }
        });
    }

    let selected_devices = platform
        .gpus
        .iter()
        .filter(|gpu| match resolved {
            AcceleratorClass::Cuda => gpu.backend.as_deref() == Some("nvidia"),
            AcceleratorClass::Metal => gpu.backend.as_deref() == Some("metal"),
            _ => false,
        })
        .map(|gpu| AcceleratorDevice {
            id: gpu
                .id
                .clone()
                .or_else(|| gpu.index.map(|index| index.to_string())),
            name: gpu.model.clone(),
            backend: resolved,
        })
        .collect::<Vec<_>>();

    let mut driver_versions = BTreeMap::new();
    for (key, value) in &platform.accelerator_metadata {
        if key.contains("version") || key == "collector" || key == "unified_memory" {
            driver_versions.insert(key.clone(), value.clone());
        }
    }

    let use_proof_source = if runtime_forced_cpu {
        if runtime_forces_cpu() {
            "env:motlie_model_force_cpu".to_owned()
        } else {
            "env:motlie_model_gpu_layers".to_owned()
        }
    } else if caller_reported_backend {
        "backend_observation".to_owned()
    } else {
        match resolved {
            AcceleratorClass::Cuda | AcceleratorClass::Metal => "backend:unreported".to_owned(),
            AcceleratorClass::Cpu => "profile:cpu".to_owned(),
            AcceleratorClass::Any | AcceleratorClass::Unavailable => "unavailable".to_owned(),
        }
    };

    AcceleratorSection {
        requested_class: requested,
        resolved_class: resolved,
        selected_devices,
        backend_mode,
        offload,
        driver_versions,
        fallback_reason,
        use_proof_source: Some(use_proof_source),
    }
}

pub fn evaluate_use(accelerator: &AcceleratorSection) -> AcceptanceStatus {
    if accelerator.fallback_reason == Some(OutcomeReason::BackendOffloadUnverified) {
        return AcceptanceStatus::Blocked;
    }

    match accelerator.requested_class {
        AcceleratorClass::Any | AcceleratorClass::Cpu => AcceptanceStatus::Pass,
        requested if requested == accelerator.resolved_class => AcceptanceStatus::Pass,
        _ => AcceptanceStatus::Blocked,
    }
}

fn runtime_forces_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn runtime_gpu_layers_override() -> Option<u32> {
    std::env::var("MOTLIE_MODEL_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
}

pub fn runtime_gpu_layers() -> Option<u32> {
    if runtime_forces_cpu() {
        Some(0)
    } else {
        runtime_gpu_layers_override()
    }
}

fn runtime_offload_override() -> Option<String> {
    runtime_gpu_layers().map(|layers| format!("gpu_layers={layers}"))
}

fn available_class(platform: &PlatformSnapshot) -> AcceleratorClass {
    match platform.gpu_backend.as_deref() {
        Some("nvidia") => AcceleratorClass::Cuda,
        Some("metal") => AcceleratorClass::Metal,
        _ => AcceleratorClass::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{GpuSnapshot, PlatformSnapshot};

    #[test]
    fn profile_requests_cuda() {
        assert_eq!(requested_for_profile("dgx-spark"), AcceleratorClass::Cuda);
        assert_eq!(
            requested_for_profile("cuda-workstation"),
            AcceleratorClass::Cuda
        );
    }

    #[test]
    fn inventory_only_metal_is_unverified_until_backend_reports_device() {
        let platform = platform_with("metal", "Apple M3");
        let accelerator = resolve_for_profile("apple-metal", &platform);

        assert_eq!(accelerator.requested_class, AcceleratorClass::Metal);
        assert_eq!(accelerator.resolved_class, AcceleratorClass::Metal);
        assert_eq!(accelerator.selected_devices.len(), 1);
        assert_eq!(
            accelerator.fallback_reason,
            Some(OutcomeReason::BackendOffloadUnverified)
        );
        assert_eq!(
            accelerator.use_proof_source.as_deref(),
            Some("backend:unreported")
        );
        assert_eq!(evaluate_use(&accelerator), AcceptanceStatus::Blocked);
    }

    #[test]
    fn backend_observation_credits_metal_use() {
        let platform = platform_with("metal", "Apple M3");
        let accelerator = resolve(
            AcceleratorClass::Metal,
            &platform,
            Some("metal".to_owned()),
            Some("selected_device=0".to_owned()),
        );

        assert_eq!(accelerator.resolved_class, AcceleratorClass::Metal);
        assert_eq!(accelerator.fallback_reason, None);
        assert_eq!(
            accelerator.use_proof_source.as_deref(),
            Some("backend_observation")
        );
        assert_eq!(evaluate_use(&accelerator), AcceptanceStatus::Pass);
    }

    #[test]
    fn cuda_profile_blocks_without_cuda_inventory() {
        let platform = PlatformSnapshot {
            gpu_backend: Some("unavailable".to_owned()),
            ..empty_platform()
        };
        let accelerator = resolve_for_profile("dgx-spark", &platform);

        assert_eq!(accelerator.requested_class, AcceleratorClass::Cuda);
        assert_eq!(accelerator.resolved_class, AcceleratorClass::Unavailable);
        assert_eq!(
            accelerator.fallback_reason,
            Some(OutcomeReason::AcceleratorUnavailable)
        );
        assert_eq!(evaluate_use(&accelerator), AcceptanceStatus::Blocked);
    }

    fn platform_with(backend: &str, model: &str) -> PlatformSnapshot {
        PlatformSnapshot {
            gpu_backend: Some(backend.to_owned()),
            gpus: vec![GpuSnapshot {
                index: Some(0),
                backend: Some(backend.to_owned()),
                id: Some("0".to_owned()),
                model: Some(model.to_owned()),
                total_memory_bytes: None,
                free_memory_bytes: None,
                unified_memory: Some(backend == "metal"),
                recommended_max_working_set_size_bytes: None,
            }],
            ..empty_platform()
        }
    }

    fn empty_platform() -> PlatformSnapshot {
        PlatformSnapshot {
            os: Some("linux".to_owned()),
            arch: Some("x86_64".to_owned()),
            target_triple: None,
            hostname: Some("host".to_owned()),
            host_id: Some("host".to_owned()),
            host_slug: Some("host".to_owned()),
            total_memory_bytes: None,
            available_memory_bytes: None,
            total_swap_bytes: None,
            free_swap_bytes: None,
            gpu_backend: None,
            gpus: Vec::new(),
            accelerator_metadata: BTreeMap::new(),
            unavailable: Vec::new(),
        }
    }
}
