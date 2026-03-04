use anyhow::{bail, Result};
use motlie_db::storage_builder::StorageBuilder;
use motlie_db::{fulltext, graph, vector, Id};

fn main() -> Result<()> {
    let base_path = std::env::temp_dir().join(format!(
        "motlie-db-storage-builder-example-{}",
        Id::new()
    ));

    let storage = StorageBuilder::new(&base_path)
        .with_rocksdb(Box::new(graph::Subsystem::new()))
        .with_rocksdb(Box::new(vector::Subsystem::new()))
        .with_fulltext(Box::new(fulltext::Schema::new()))
        .with_cache_size(256 * 1024 * 1024)
        .build()?;

    if !storage.has_rocksdb() {
        bail!("expected RocksDB storage to be initialized");
    }
    if !storage.has_tantivy() {
        bail!("expected Tantivy storage to be initialized");
    }

    println!("base path: {}", storage.path().display());
    println!("rocksdb path: {}", storage.rocksdb_path().display());
    println!("tantivy path: {}", storage.tantivy_path().display());
    println!("rocksdb components: {:?}", storage.component_names());
    println!("fulltext components: {:?}", storage.fulltext_names());
    println!("total column families: {}", storage.all_cf_names().len());

    if let Err(err) = std::fs::remove_dir_all(&base_path) {
        eprintln!(
            "warning: failed to remove temporary path {}: {}",
            base_path.display(),
            err
        );
    }

    Ok(())
}
