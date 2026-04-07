use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use motlie_vmm::backend::BackendKind;
use motlie_vmm::network::{AdminNetMode, EgressNetMode, NetworkModes};
use motlie_vmm::network_alloc::{GuestNetAllocator, GuestNetAllocatorConfig};
use motlie_vmm::orchestrator::{PrepareRequest, ReadinessPolicy, VmHandle, boot, prepare};
use motlie_vmm::spec::{
    BootArtifacts, GuestResources, GuestSpec, GuestSshAccess, GuestStorage, GuestUser,
    RuntimeNamespace, SoftwareProfile,
};

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), DynError> {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/v1.4");
    let boot_artifacts = BootArtifacts {
        kernel: base_dir.join("artifacts/base/Image"),
        initramfs: None,
        firmware: None,
        cmdline: None,
    };
    let namespace = RuntimeNamespace::new("motlie-vmm-v14", "/tmp")?;
    let mut allocator = GuestNetAllocator::new(GuestNetAllocatorConfig {
        socket_dir: namespace.temp_root.join(format!("{}-sockets", namespace.prefix)),
        ..GuestNetAllocatorConfig::default()
    });
    let mut handles: HashMap<String, VmHandle> = HashMap::new();

    println!("=== motlie-vmm repl_host_v1_4 ===");
    println!("Thin harness over libs/vmm phase-3 services");
    println!("Commands: boot <guest> | ready <guest> | shutdown <guest> | status | quit");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    print!("v14> ");
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("boot") => {
                let Some(guest_id) = parts.next() else {
                    println!("error: boot <guest>");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                };
                if handles.contains_key(guest_id) {
                    println!("error: guest '{guest_id}' already booted");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                }

                let request = PrepareRequest {
                    guest: demo_guest(guest_id, &boot_artifacts),
                    namespace: namespace.clone(),
                    network_modes: NetworkModes {
                        admin: AdminNetMode::None,
                        egress: EgressNetMode::None,
                    },
                    backend_kind: BackendKind::ChShell,
                    base_dir: base_dir.clone(),
                    ssh_ca_pubkey: None,
                };
                let prepared = prepare(request, &mut allocator)?;
                let handle = boot(prepared)?;
                println!(
                    "ok: booted {} pid={:?} api={}",
                    guest_id,
                    handle.pid,
                    handle.runtime_paths.api_socket.display()
                );
                handles.insert(guest_id.to_string(), handle);
            }
            Some("ready") => {
                let Some(guest_id) = parts.next() else {
                    println!("error: ready <guest>");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                };
                let Some(handle) = handles.get(guest_id) else {
                    println!("error: unknown guest '{guest_id}'");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                };
                handle.ready(&ReadinessPolicy::default()).await?;
                println!("ok: {guest_id} api socket ready");
            }
            Some("shutdown") => {
                let Some(guest_id) = parts.next() else {
                    println!("error: shutdown <guest>");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                };
                let Some(handle) = handles.remove(guest_id) else {
                    println!("error: unknown guest '{guest_id}'");
                    print!("v14> ");
                    stdout.flush()?;
                    continue;
                };
                let report = handle.shutdown().await?;
                println!(
                    "ok: shutdown {} pid={:?} forced={:?}",
                    guest_id, report.pid, report.forced
                );
            }
            Some("status") => {
                if handles.is_empty() {
                    println!("(no guests)");
                } else {
                    for (guest_id, handle) in &handles {
                        println!(
                            "{} pid={:?} api={} vnet={}",
                            guest_id,
                            handle.pid,
                            handle.runtime_paths.api_socket.display(),
                            handle.runtime_paths.vnet_socket.display()
                        );
                    }
                }
            }
            Some("quit") | Some("exit") => break,
            Some("") | None => {}
            Some(cmd) => println!("error: unknown command '{cmd}'"),
        }

        print!("v14> ");
        stdout.flush()?;
    }

    Ok(())
}

fn demo_guest(guest_id: &str, boot: &BootArtifacts) -> GuestSpec {
    GuestSpec {
        guest_id: guest_id.to_string(),
        hostname: format!("motlie-{guest_id}"),
        socket_path: format!("/tmp/motlie-vmm-v14-{guest_id}.vsock_5000"),
        user: GuestUser {
            name: guest_id.to_string(),
            uid: 1000,
            gid: 1000,
            home: PathBuf::from(format!("/home/{guest_id}")),
        },
        ssh: GuestSshAccess {
            principal: guest_id.to_string(),
            login_user: guest_id.to_string(),
        },
        mounts: vec![],
        software: SoftwareProfile::default(),
        resources: GuestResources::default(),
        storage: GuestStorage::default(),
        boot: boot.clone(),
    }
}
