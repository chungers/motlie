//! Build script for SIMD feature detection
//!
//! Detects target platform and sets appropriate cfg flags:
//! - `simd_level="avx512"` - x86_64 with AVX-512 (DGX Spark)
//! - `simd_level="avx2"` - x86_64 with AVX2+FMA (most servers)
//! - `simd_level="neon"` - aarch64 (macOS Apple Silicon)
//! - `simd_level="runtime"` - Runtime detection
//! - `simd_level="scalar"` - Fallback
//!
//! ## Usage
//!
//! ```bash
//! # macOS Apple Silicon (auto-detects NEON)
//! cargo build --release
//!
//! # DGX Spark (AVX-512)
//! cargo build --release --features simd-avx512
//!
//! # Portable binary with runtime detection
//! cargo build --release --features simd-runtime
//!
//! # Maximum performance for current CPU
//! RUSTFLAGS='-C target-cpu=native' cargo build --release --features simd-native
//! ```

use std::env;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    // Declare expected cfg values for simd_level (suppresses warnings)
    println!("cargo::rustc-check-cfg=cfg(simd_level, values(\"avx512\", \"avx2\", \"neon\", \"scalar\", \"runtime\"))");
    println!("cargo::rustc-check-cfg=cfg(platform, values(\"dgx\", \"apple_silicon\"))");

    // Rerun if these change
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");

    // Check for explicit feature flags
    let simd_native = env::var("CARGO_FEATURE_SIMD_NATIVE").is_ok();
    let simd_avx512 = env::var("CARGO_FEATURE_SIMD_AVX512").is_ok();
    let simd_avx2 = env::var("CARGO_FEATURE_SIMD_AVX2").is_ok();
    let simd_neon = env::var("CARGO_FEATURE_SIMD_NEON").is_ok();
    let simd_none = env::var("CARGO_FEATURE_SIMD_NONE").is_ok();
    let simd_runtime = env::var("CARGO_FEATURE_SIMD_RUNTIME").is_ok();

    // Priority: explicit flags > auto-detection

    // 1. Explicit scalar fallback
    if simd_none {
        set_simd_level("scalar", "Using scalar fallback (simd-none feature)");
        return;
    }

    // 2. Explicit runtime detection
    if simd_runtime && !simd_avx512 && !simd_avx2 && !simd_neon {
        set_simd_level("runtime", "Using runtime CPU detection (simd-runtime feature)");
        return;
    }

    // 3. Explicit AVX-512 (DGX Spark)
    if simd_avx512 {
        set_simd_level("avx512", "Targeting AVX-512 (simd-avx512 feature)");
        set_platform_hint("dgx");
        return;
    }

    // 4. Explicit AVX2
    if simd_avx2 {
        set_simd_level("avx2", "Targeting AVX2+FMA (simd-avx2 feature)");
        return;
    }

    // 5. Explicit NEON
    if simd_neon {
        set_simd_level("neon", "Targeting NEON (simd-neon feature)");
        return;
    }

    // 6. Auto-detect based on target architecture
    match target_arch.as_str() {
        "x86_64" => detect_x86_64(&target_features, simd_native),
        "aarch64" => detect_aarch64(&target_os),
        arch => {
            set_simd_level("scalar", &format!("Unknown arch '{}', using scalar", arch));
        }
    }
}

fn detect_x86_64(target_features: &str, simd_native: bool) {
    // Check target features (set via RUSTFLAGS or --target-feature)
    let has_avx512 = target_features.contains("avx512f");
    let has_avx2 = target_features.contains("avx2");
    let has_fma = target_features.contains("fma");

    if has_avx512 {
        set_simd_level("avx512", "Auto-detected AVX-512 for x86_64");
        set_platform_hint("dgx");
    } else if has_avx2 && has_fma {
        set_simd_level("avx2", "Auto-detected AVX2+FMA for x86_64");
    } else if simd_native {
        // When -C target-cpu=native is used, features may not be in CARGO_CFG_TARGET_FEATURE
        // but the compiler will use them. Default to runtime detection to be safe.
        set_simd_level("runtime", "Using runtime detection with -C target-cpu=native");
    } else {
        // No explicit features, use runtime detection
        set_simd_level("runtime", "Using runtime detection for x86_64");
    }
}

fn detect_aarch64(target_os: &str) {
    // NEON is always available on aarch64
    if target_os == "macos" {
        set_simd_level("neon", "Targeting NEON for macOS Apple Silicon");
        set_platform_hint("apple_silicon");
    } else {
        set_simd_level("neon", "Targeting NEON for aarch64 Linux");
    }
}

fn set_simd_level(level: &str, message: &str) {
    println!("cargo:rustc-cfg=simd_level=\"{}\"", level);
    println!("cargo:warning=SIMD: {}", message);
}

fn set_platform_hint(platform: &str) {
    println!("cargo:rustc-cfg=platform=\"{}\"", platform);
}
