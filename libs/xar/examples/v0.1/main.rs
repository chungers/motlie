//! `xar` v0.1 skeleton: GGUF round-trip only.
//!
//! This example is intentionally a workflow stub rather than an implementation.
//! It records the first end-to-end slice `libs/xar` is designed to validate.

fn main() {
    let steps = [
        "1. Select a local GGUF weight file as the phase-1 payload.",
        "2. Pack it into an OCI-backed xar payload with a Motlie preamble TOC.",
        "3. Append the payload to an executable and record footer -> preamble offsets.",
        "4. At runtime, read footer and preamble to resolve the GGUF blob by digest.",
        "5. Extract to a local root for v0.1 and hand off via libs/model ArtifactPolicy::LocalOnly.",
        "6. In a later slice, replace extraction with direct mmap using (fd, offset, len).",
    ];

    println!("xar v0.1 skeleton");
    for step in steps {
        println!("{step}");
    }
}
