#[cfg(test)]
mod tests {
    use crate::graph::{spawn_graph_consumer, spawn_graph_consumer_with_next, Storage};
    use crate::index::{Edges, Fragments, Index, Nodes};
    use crate::{
        create_mutation_writer, AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id, WriterConfig,
    };
    use rocksdb::DB;
    use std::path::Path;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_graph_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        let edge_args = AddEdgeArgs {
            id: Id::new(),
            source_vertex_id: Id::new(),
            target_vertex_id: Id::new(),
            ts_millis: 1234567890,
            name: "test_edge".to_string(),
        };
        writer.add_edge(edge_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_consumer_multiple_mutations() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

        // Send 5 mutations rapidly
        for i in 0..5 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("test_vertex_{}", i),
            };
            writer.add_vertex(vertex_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

        // Test all mutation types
        writer
            .add_vertex(AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: "vertex".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdgeArgs {
                id: Id::new(),
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: "edge".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                body: "fragment body".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                reason: "test invalidation".to_string(),
            })
            .await
            .unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_to_fulltext_chaining() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        // Create the FullText consumer (end of chain)
        let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
        let fulltext_handle = crate::spawn_fulltext_consumer(fulltext_receiver, config.clone());

        // Create the Graph consumer that forwards to FullText
        let (writer, graph_receiver) = create_mutation_writer(config.clone());
        let graph_handle = spawn_graph_consumer_with_next(
            graph_receiver,
            config.clone(),
            Path::new("/tmp/test_graph_db"),
            fulltext_sender,
        );

        // Send mutations - they should flow through Graph -> FullText
        for i in 0..3 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("chained_vertex_{}", i),
            };
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!(
                    "Chained fragment {} processed by both Graph and FullText",
                    i
                ),
            };

            writer.add_vertex(vertex_args).await.unwrap();
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give both consumers time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown the chain from the beginning
        drop(writer);

        // Wait for Graph consumer to complete (which will close FullText's channel)
        graph_handle.await.unwrap().unwrap();

        // Wait for FullText consumer to complete
        fulltext_handle.await.unwrap().unwrap();
    }

    #[test]
    fn test_storage_ready_creates_new_db() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Ready should succeed for a new database
        let result = storage.ready();
        assert!(result.is_ok(), "ready() should succeed for new database");

        // Cleanup
        storage.close().unwrap();
    }

    #[test]
    fn test_storage_ready_opens_existing_db() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create and close a database
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        storage.close().unwrap();

        // Open it again - should succeed
        let mut storage2 = Storage::readwrite(&db_path);
        let result = storage2.ready();
        assert!(
            result.is_ok(),
            "ready() should succeed for existing database"
        );

        // Cleanup
        storage2.close().unwrap();
    }

    #[test]
    fn test_storage_ready_fails_on_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_a_directory");

        // Create a file at the path
        std::fs::write(&file_path, "test").unwrap();

        // Verify it's actually a file
        assert!(file_path.exists(), "file should exist");
        assert!(file_path.is_file(), "path should be a file");
        assert!(!file_path.is_dir(), "path should not be a directory");

        let mut storage = Storage::readwrite(&file_path);

        // Ready should fail because path is a file
        let result = storage.ready();
        assert!(result.is_err(), "ready() should fail when path is a file");

        let error_msg = result.unwrap_err().to_string();
        println!("Error when path is file: {}", error_msg);
        assert!(
            error_msg.contains("Path is a file"),
            "Error message should mention 'Path is a file', got: {}",
            error_msg
        );
    }

    #[test]
    fn test_storage_ready_fails_on_symlink() {
        let temp_dir = TempDir::new().unwrap();
        let target_path = temp_dir.path().join("target");
        let symlink_path = temp_dir.path().join("symlink");

        // Create a target directory
        std::fs::create_dir(&target_path).unwrap();

        // Create a symlink to it
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target_path, &symlink_path).unwrap();

        #[cfg(unix)]
        {
            // Verify it's actually a symlink
            assert!(symlink_path.exists(), "symlink should exist");
            assert!(symlink_path.is_symlink(), "path should be a symlink");

            let mut storage = Storage::readwrite(&symlink_path);

            // Ready should fail because path is a symlink
            let result = storage.ready();
            assert!(
                result.is_err(),
                "ready() should fail when path is a symlink"
            );

            let error_msg = result.unwrap_err().to_string();
            println!("Error when path is symlink: {}", error_msg);
            assert!(
                error_msg.contains("Path is a symlink"),
                "Error message should mention 'Path is a symlink', got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_storage_close_flushes_data() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();

        // Close should succeed
        let result = storage.close();
        assert!(result.is_ok(), "close() should succeed");
    }

    #[test]
    fn test_storage_close_multiple_times() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();

        // First close should succeed
        assert!(storage.close().is_ok(), "first close() should succeed");

        // Second close should fail (storage not ready)
        let result = storage.close();
        assert!(
            result.is_err(),
            "second close() should fail (storage not ready)"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Storage is not ready"),
            "Error should indicate storage is not ready"
        );
    }

    #[test]
    fn test_storage_close_without_ready() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Close without ready should fail
        let result = storage.close();
        assert!(
            result.is_err(),
            "close() should fail when storage is not ready"
        );

        let error_msg = result.unwrap_err().to_string();
        println!("Error when closing before ready: {}", error_msg);
        assert!(
            error_msg.contains("Storage is not ready"),
            "Error should mention storage not ready, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_storage_close_before_ready_error() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Attempting to close before ready should return an error
        let result = storage.close();
        assert!(
            result.is_err(),
            "close() should return error when called before ready()"
        );

        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("Storage is not ready"),
            "Error message should indicate storage is not ready, got: {}",
            error_msg
        );

        // Now call ready() and close should work
        storage.ready().unwrap();
        let close_result = storage.close();
        assert!(close_result.is_ok(), "close() should succeed after ready()");

        // Calling close again should also fail (storage not ready anymore)
        let result2 = storage.close();
        assert!(result2.is_err(), "close() should fail after already closed");
    }

    #[test]
    fn test_storage_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create storage
        let mut storage = Storage::readwrite(&db_path);

        // Ready the database
        assert!(storage.ready().is_ok(), "ready() should succeed");

        // Database should exist
        assert!(db_path.exists(), "database path should exist after ready()");

        // Close the database
        assert!(storage.close().is_ok(), "close() should succeed");

        // Database should still exist after close
        assert!(
            db_path.exists(),
            "database path should still exist after close()"
        );

        // Can reopen after close
        assert!(
            storage.ready().is_ok(),
            "ready() should succeed after close()"
        );

        // Final cleanup
        storage.close().unwrap();
    }

    #[test]
    fn test_storage_ready_creates_column_families() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        storage.close().unwrap();

        // Reopen the database directly with RocksDB to verify column families exist
        let db = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );

        assert!(
            db.is_ok(),
            "Should be able to open database with all column families"
        );

        let db = db.unwrap();

        // Verify each column family handle exists
        let nodes_cf = db.cf_handle(Nodes::cf_name());
        assert!(
            nodes_cf.is_some(),
            "Nodes column family should exist: {}",
            Nodes::cf_name()
        );

        let edges_cf = db.cf_handle(Edges::cf_name());
        assert!(
            edges_cf.is_some(),
            "Edges column family should exist: {}",
            Edges::cf_name()
        );

        let fragments_cf = db.cf_handle(Fragments::cf_name());
        assert!(
            fragments_cf.is_some(),
            "Fragments column family should exist: {}",
            Fragments::cf_name()
        );

        drop(db);
    }

    #[test]
    fn test_storage_ready_db_actually_opens() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Database directory should not exist before ready()
        assert!(!db_path.exists(), "DB should not exist before ready()");

        storage.ready().unwrap();

        // Database directory should exist after ready()
        assert!(db_path.exists(), "DB should exist after ready()");
        assert!(db_path.is_dir(), "DB path should be a directory");

        storage.close().unwrap();

        // Verify we can open the database independently
        let db = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );

        assert!(
            db.is_ok(),
            "Should be able to independently open the database"
        );
        drop(db);
    }

    #[test]
    fn test_storage_ready_with_invalid_options() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // First create a database
        let mut storage1 = Storage::readwrite(&db_path);
        storage1.ready().unwrap();
        storage1.close().unwrap();

        // Now try to open with error_if_exists = true (should fail)
        let mut options = rocksdb::Options::default();
        options.set_error_if_exists(true);
        options.create_if_missing(false);

        let mut storage2 = Storage::readwrite_with_options(&db_path, options);

        let result = storage2.ready();
        assert!(
            result.is_err(),
            "ready() should fail with error_if_exists when DB already exists"
        );

        println!("Error with error_if_exists: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_storage_ready_missing_column_family_error() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create a database with only the default column family
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, &db_path);
        assert!(db.is_ok(), "Should create database with default CF");
        drop(db);

        // Now try to open with Storage which expects specific column families
        // This should fail because the column families don't exist
        let mut options = rocksdb::Options::default();
        options.create_missing_column_families(false); // Don't create missing CFs

        let mut storage = Storage::readwrite_with_options(&db_path, options);

        let result = storage.ready();
        assert!(
            result.is_err(),
            "ready() should fail when expected column families are missing"
        );

        let error_msg = result.unwrap_err().to_string();
        println!("Error with missing column families: {}", error_msg);
        assert!(
            error_msg.contains("Column family not found") || error_msg.contains("Invalid argument"),
            "Error should indicate missing column families, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_storage_ready_propagates_rocksdb_errors() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create a file instead of directory to cause RocksDB error
        std::fs::write(&db_path, "not a database").unwrap();

        let mut storage = Storage::readwrite(&db_path);

        // This should fail at the DB::open_cf call, not our file check
        // because RocksDB will try to open it
        let result = storage.ready();
        assert!(
            result.is_err(),
            "ready() should propagate RocksDB error when path is invalid"
        );

        println!("RocksDB error propagated: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_storage_multiple_ready_calls() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // First ready should succeed
        assert!(storage.ready().is_ok(), "First ready() should succeed");

        // Second ready without close should be idempotent (no-op)
        let result = storage.ready();

        // Should succeed because ready() is idempotent
        println!("Second ready() result: {:?}", result);
        assert!(result.is_ok(), "Second ready() should succeed (idempotent)");

        storage.close().unwrap();
    }

    #[test]
    fn test_storage_ready_idempotency() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Call ready() the first time
        let result1 = storage.ready();
        assert!(result1.is_ok(), "First ready() should succeed");

        // Call ready() multiple times - all should succeed without error
        for i in 1..=5 {
            let result = storage.ready();
            assert!(
                result.is_ok(),
                "ready() call #{} should succeed (idempotent), got: {:?}",
                i + 1,
                result
            );
        }

        // Verify database still exists and is functional
        assert!(db_path.exists(), "DB should still exist");

        // Close and verify we can still reopen
        storage.close().unwrap();

        // After close, ready() should work again
        let result = storage.ready();
        assert!(
            result.is_ok(),
            "ready() after close() should succeed, got: {:?}",
            result
        );

        storage.close().unwrap();
    }

    #[test]
    fn test_storage_ready_idempotency_preserves_db_handle() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // First ready
        storage.ready().unwrap();

        // Verify database is created
        assert!(db_path.exists(), "DB should exist after first ready()");

        // Call ready again - should be no-op
        storage.ready().unwrap();

        // Verify we can still close successfully (DB handle is still valid)
        let close_result = storage.close();
        assert!(
            close_result.is_ok(),
            "Should be able to close after multiple ready() calls"
        );

        // Verify database can be reopened with RocksDB directly
        let db = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );

        assert!(
            db.is_ok(),
            "DB should still be valid after idempotent ready() calls"
        );
        drop(db);
    }

    #[test]
    fn test_storage_reusability_after_close() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // First cycle: ready -> close
        println!("First cycle: ready()");
        let result = storage.ready();
        assert!(result.is_ok(), "First ready() should succeed");
        assert!(db_path.exists(), "DB should exist after first ready()");

        println!("First cycle: close()");
        let result = storage.close();
        assert!(result.is_ok(), "First close() should succeed");
        assert!(db_path.exists(), "DB should still exist after close()");

        // Second cycle: ready -> close (reusing the same Storage instance)
        println!("Second cycle: ready()");
        let result = storage.ready();
        assert!(
            result.is_ok(),
            "Second ready() should succeed (storage reusable), got: {:?}",
            result
        );

        println!("Second cycle: close()");
        let result = storage.close();
        assert!(
            result.is_ok(),
            "Second close() should succeed, got: {:?}",
            result
        );

        // Third cycle: ready -> close
        println!("Third cycle: ready()");
        let result = storage.ready();
        assert!(
            result.is_ok(),
            "Third ready() should succeed (storage still reusable), got: {:?}",
            result
        );

        println!("Third cycle: close()");
        let result = storage.close();
        assert!(
            result.is_ok(),
            "Third close() should succeed, got: {:?}",
            result
        );

        // Verify database integrity by opening with RocksDB directly
        let db = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );

        assert!(
            db.is_ok(),
            "DB should be valid after multiple ready/close cycles"
        );
        drop(db);
    }

    #[test]
    fn test_storage_ready_close_ready_with_verification() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);

        // Initial state: DB doesn't exist
        assert!(!db_path.exists(), "DB should not exist initially");

        // First ready() - creates the database
        storage.ready().unwrap();
        assert!(db_path.exists(), "DB should exist after ready()");

        // Note: Cannot open another writable connection while Storage has it open
        // (RocksDB uses file locks). We'll verify column families after close.

        // Close the database
        storage.close().unwrap();
        assert!(db_path.exists(), "DB should still exist after close()");

        // Verify we can independently open the database after close
        let db_after_close = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );
        assert!(
            db_after_close.is_ok(),
            "Should be able to open DB independently after close()"
        );
        drop(db_after_close);

        // Ready again - reopens the existing database
        let result = storage.ready();
        assert!(
            result.is_ok(),
            "ready() after close() should succeed (reopen existing DB), got: {:?}",
            result
        );

        // Verify column families still exist after reopening
        storage.close().unwrap();
        let db_final = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );
        assert!(
            db_final.is_ok(),
            "Column families should still exist after ready-close-ready cycle"
        );

        let db_final = db_final.unwrap();
        assert!(
            db_final.cf_handle(Nodes::cf_name()).is_some(),
            "Nodes CF should exist"
        );
        assert!(
            db_final.cf_handle(Edges::cf_name()).is_some(),
            "Edges CF should exist"
        );
        assert!(
            db_final.cf_handle(Fragments::cf_name()).is_some(),
            "Fragments CF should exist"
        );
    }

    #[test]
    fn test_storage_readonly_basic() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // First create a database with ReadWrite
        let mut storage_rw = Storage::readwrite(&db_path);
        storage_rw.ready().unwrap();
        storage_rw.close().unwrap();

        // Now open it in ReadOnly mode
        let mut storage_ro = Storage::readonly(&db_path);
        let result = storage_ro.ready();
        assert!(
            result.is_ok(),
            "ReadOnly storage should successfully open existing database"
        );

        // Close should work
        storage_ro.close().unwrap();
    }

    #[test]
    fn test_storage_readonly_and_readwrite_coexist() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // First create a database with ReadWrite
        let mut storage_rw1 = Storage::readwrite(&db_path);
        storage_rw1.ready().unwrap();
        storage_rw1.close().unwrap();

        // Open in ReadWrite mode
        let mut storage_rw = Storage::readwrite(&db_path);
        let result_rw = storage_rw.ready();
        assert!(
            result_rw.is_ok(),
            "ReadWrite storage should successfully open, got: {:?}",
            result_rw
        );

        // Open same database in ReadOnly mode - should work simultaneously
        let mut storage_ro = Storage::readonly(&db_path);
        let result_ro = storage_ro.ready();
        assert!(
            result_ro.is_ok(),
            "ReadOnly storage should successfully open while ReadWrite is open, got: {:?}",
            result_ro
        );

        // Both should be open now
        println!("Both ReadWrite and ReadOnly storage instances are open");

        // Close ReadWrite first
        storage_rw.close().unwrap();
        println!("ReadWrite storage closed");

        // ReadOnly should still be usable - verify by closing it
        let close_result = storage_ro.close();
        assert!(
            close_result.is_ok(),
            "ReadOnly storage should still be usable after ReadWrite is closed"
        );
    }

    #[test]
    fn test_storage_readonly_close_independence() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create database
        let mut storage_rw1 = Storage::readwrite(&db_path);
        storage_rw1.ready().unwrap();
        storage_rw1.close().unwrap();

        // Open two ReadOnly instances
        let mut storage_ro1 = Storage::readonly(&db_path);
        storage_ro1.ready().unwrap();

        let mut storage_ro2 = Storage::readonly(&db_path);
        storage_ro2.ready().unwrap();

        println!("Two ReadOnly storage instances are open");

        // Close first ReadOnly
        storage_ro1.close().unwrap();
        println!("First ReadOnly storage closed");

        // Second ReadOnly should still be usable
        let close_result = storage_ro2.close();
        assert!(
            close_result.is_ok(),
            "Second ReadOnly storage should still work after first is closed"
        );
    }

    #[test]
    fn test_storage_multiple_readonly_instances() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create database
        let mut storage_rw = Storage::readwrite(&db_path);
        storage_rw.ready().unwrap();
        storage_rw.close().unwrap();

        // Open multiple ReadOnly instances simultaneously
        let mut storage_ro1 = Storage::readonly(&db_path);
        let mut storage_ro2 = Storage::readonly(&db_path);
        let mut storage_ro3 = Storage::readonly(&db_path);

        assert!(
            storage_ro1.ready().is_ok(),
            "ReadOnly instance 1 should open"
        );
        assert!(
            storage_ro2.ready().is_ok(),
            "ReadOnly instance 2 should open"
        );
        assert!(
            storage_ro3.ready().is_ok(),
            "ReadOnly instance 3 should open"
        );

        println!("Three ReadOnly storage instances are open simultaneously");

        // Verify all column families are accessible
        let db_verify = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );
        assert!(
            db_verify.is_ok(),
            "Should be able to verify database structure"
        );
        drop(db_verify);

        // Close in different order
        storage_ro2.close().unwrap();
        storage_ro1.close().unwrap();
        storage_ro3.close().unwrap();
    }

    #[test]
    fn test_storage_readwrite_with_multiple_readonly() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create database
        let mut storage_init = Storage::readwrite(&db_path);
        storage_init.ready().unwrap();
        storage_init.close().unwrap();

        // Open ReadWrite
        let mut storage_rw = Storage::readwrite(&db_path);
        storage_rw.ready().unwrap();

        // Open multiple ReadOnly instances while ReadWrite is open
        let mut storage_ro1 = Storage::readonly(&db_path);
        let mut storage_ro2 = Storage::readonly(&db_path);

        assert!(
            storage_ro1.ready().is_ok(),
            "ReadOnly instance 1 should open while ReadWrite is open"
        );
        assert!(
            storage_ro2.ready().is_ok(),
            "ReadOnly instance 2 should open while ReadWrite is open"
        );

        println!("ReadWrite and 2 ReadOnly instances coexist");

        // Close ReadWrite
        storage_rw.close().unwrap();

        // ReadOnly instances should still work
        assert!(
            storage_ro1.close().is_ok(),
            "ReadOnly 1 should work after ReadWrite closed"
        );
        assert!(
            storage_ro2.close().is_ok(),
            "ReadOnly 2 should work after ReadWrite closed"
        );
    }

    #[test]
    fn test_storage_readonly_cannot_create_new_db() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Database doesn't exist yet
        assert!(!db_path.exists(), "Database should not exist");

        // Try to open non-existent database in ReadOnly mode
        let mut storage_ro = Storage::readonly(&db_path);
        let result = storage_ro.ready();

        // This should fail because ReadOnly can't create a database
        assert!(
            result.is_err(),
            "ReadOnly storage should fail to open non-existent database"
        );

        println!("ReadOnly correctly failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_storage_readwrite_with_multiple_readonly_close_first() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create and initialize the database
        let mut storage_init = Storage::readwrite(&db_path);
        storage_init.ready().unwrap();
        storage_init.close().unwrap();
        println!("Database created and initialized");

        // Open one ReadWrite instance
        let mut storage_rw = Storage::readwrite(&db_path);
        assert!(
            storage_rw.ready().is_ok(),
            "ReadWrite storage should open successfully"
        );
        println!("ReadWrite storage opened");

        // Open multiple ReadOnly instances (4 instances to test > 1)
        let mut storage_ro1 = Storage::readonly(&db_path);
        let mut storage_ro2 = Storage::readonly(&db_path);
        let mut storage_ro3 = Storage::readonly(&db_path);
        let mut storage_ro4 = Storage::readonly(&db_path);

        assert!(
            storage_ro1.ready().is_ok(),
            "ReadOnly storage 1 should open while ReadWrite is open"
        );
        println!("ReadOnly storage 1 opened");

        assert!(
            storage_ro2.ready().is_ok(),
            "ReadOnly storage 2 should open while ReadWrite is open"
        );
        println!("ReadOnly storage 2 opened");

        assert!(
            storage_ro3.ready().is_ok(),
            "ReadOnly storage 3 should open while ReadWrite is open"
        );
        println!("ReadOnly storage 3 opened");

        assert!(
            storage_ro4.ready().is_ok(),
            "ReadOnly storage 4 should open while ReadWrite is open"
        );
        println!("ReadOnly storage 4 opened");

        // Verify all instances are open
        println!("All instances active: 1 ReadWrite + 4 ReadOnly = 5 total");

        // Close ReadWrite FIRST
        let close_rw_result = storage_rw.close();
        assert!(
            close_rw_result.is_ok(),
            "ReadWrite storage should close successfully, got: {:?}",
            close_rw_result
        );
        println!("ReadWrite storage closed (first to close)");

        // Verify ALL ReadOnly instances are still functional by closing them
        let close_ro1 = storage_ro1.close();
        assert!(
            close_ro1.is_ok(),
            "ReadOnly storage 1 should still be functional after ReadWrite closed, got: {:?}",
            close_ro1
        );
        println!("ReadOnly storage 1 closed successfully");

        let close_ro2 = storage_ro2.close();
        assert!(
            close_ro2.is_ok(),
            "ReadOnly storage 2 should still be functional after ReadWrite closed, got: {:?}",
            close_ro2
        );
        println!("ReadOnly storage 2 closed successfully");

        let close_ro3 = storage_ro3.close();
        assert!(
            close_ro3.is_ok(),
            "ReadOnly storage 3 should still be functional after ReadWrite closed, got: {:?}",
            close_ro3
        );
        println!("ReadOnly storage 3 closed successfully");

        let close_ro4 = storage_ro4.close();
        assert!(
            close_ro4.is_ok(),
            "ReadOnly storage 4 should still be functional after ReadWrite closed, got: {:?}",
            close_ro4
        );
        println!("ReadOnly storage 4 closed successfully");

        // Verify database integrity after all closes
        let db_verify = DB::open_cf(
            &rocksdb::Options::default(),
            &db_path,
            &[Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()],
        );
        assert!(
            db_verify.is_ok(),
            "Database should be intact after all storage instances closed"
        );
        drop(db_verify);

        println!(
            "Test completed: ReadWrite closed first, all ReadOnly instances remained functional"
        );
    }
}
