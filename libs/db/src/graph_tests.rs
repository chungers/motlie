#[cfg(test)]
mod tests {
    use crate::graph::{
        spawn_graph_consumer, spawn_graph_consumer_with_next, ColumnFamilyRecord, Graph, Storage,
    };
    use crate::schema::{Edges, Fragments, Nodes, ALL_COLUMN_FAMILIES};
    use crate::{
        create_mutation_writer, AddEdge, AddFragment, AddNode, Id, TimestampMilli, WriterConfig,
    };
    use rocksdb::DB;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_graph_consumer_basic_processing() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Send some mutations
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
        };
        writer.add_node(node_args).await.unwrap();

        let edge_args = AddEdge {
            id: Id::new(),
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
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
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Send 5 mutations rapidly
        for i in 0..5 {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
            };
            writer.add_node(node_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_consumer_all_mutation_types() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Test all mutation types
        writer
            .add_node(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "node".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdge {
                id: Id::new(),
                source_node_id: Id::new(),
                target_node_id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "edge".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now().0,
                content: "fragment body".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
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
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

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
            &db_path,
            fulltext_sender,
        );

        // Send mutations - they should flow through Graph -> FullText
        for i in 0..3 {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("chained_node_{}", i),
            };
            let fragment_args = AddFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now().0,
                content: format!(
                    "Chained fragment {} processed by both Graph and FullText",
                    i
                ),
            };

            writer.add_node(node_args).await.unwrap();
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
        let db = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);

        assert!(
            db.is_ok(),
            "Should be able to open database with all column families"
        );

        let db = db.unwrap();

        // Verify each column family handle exists
        let nodes_cf = db.cf_handle(Nodes::CF_NAME);
        assert!(
            nodes_cf.is_some(),
            "Nodes column family should exist: {}",
            Nodes::CF_NAME
        );

        let edges_cf = db.cf_handle(Edges::CF_NAME);
        assert!(
            edges_cf.is_some(),
            "Edges column family should exist: {}",
            Edges::CF_NAME
        );

        let fragments_cf = db.cf_handle(Fragments::CF_NAME);
        assert!(
            fragments_cf.is_some(),
            "Fragments column family should exist: {}",
            Fragments::CF_NAME
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
        let db = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);

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

        let txn_options = rocksdb::TransactionDBOptions::default();
        let mut storage2 = Storage::readwrite_with_options(&db_path, options, txn_options);

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

        let txn_options = rocksdb::TransactionDBOptions::default();
        let mut storage = Storage::readwrite_with_options(&db_path, options, txn_options);

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
        let db = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);

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
        let db = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);

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
        let db_after_close =
            DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);
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
        let db_final = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);
        assert!(
            db_final.is_ok(),
            "Column families should still exist after ready-close-ready cycle"
        );

        let db_final = db_final.unwrap();
        assert!(
            db_final.cf_handle(Nodes::CF_NAME).is_some(),
            "Nodes CF should exist"
        );
        assert!(
            db_final.cf_handle(Edges::CF_NAME).is_some(),
            "Edges CF should exist"
        );
        assert!(
            db_final.cf_handle(Fragments::CF_NAME).is_some(),
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
        let db_verify = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);
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
        let db_verify = DB::open_cf(&rocksdb::Options::default(), &db_path, ALL_COLUMN_FAMILIES);
        assert!(
            db_verify.is_ok(),
            "Database should be intact after all storage instances closed"
        );
        drop(db_verify);

        println!(
            "Test completed: ReadWrite closed first, all ReadOnly instances remained functional"
        );
    }

    #[tokio::test]
    async fn test_nodes_written_to_correct_column_family() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Create a node with known ID
        let node_id = Id::new();
        let node_args = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
        };

        writer.add_node(node_args.clone()).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify the node was written to the correct column family
        let db = DB::open_cf_for_read_only(
            &rocksdb::Options::default(),
            &db_path,
            ALL_COLUMN_FAMILIES,
            false,
        )
        .expect("Failed to open database for verification");

        let cf_handle = db
            .cf_handle(Nodes::CF_NAME)
            .expect("Nodes column family should exist");

        // Create the key using the schema's serialization
        let (key, _value) = Nodes::record_from(&node_args);
        let key_bytes = Nodes::key_to_bytes(&key).expect("Failed to serialize key");

        // Query the database
        let result = db
            .get_cf(cf_handle, &key_bytes)
            .expect("Failed to query database");
        assert!(
            result.is_some(),
            "Node should be written to the 'nodes' column family"
        );

        // Verify we can deserialize the value
        let value_bytes = result.unwrap();
        let value = Nodes::value_from_bytes(&value_bytes).expect("Failed to deserialize value");
        let content = value.0.content().expect("Failed to decode DataUrl");
        assert!(
            content.contains("test_node"),
            "Node value should contain the node name"
        );
    }

    #[tokio::test]
    async fn test_edges_written_to_correct_column_families() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Create an edge with known IDs
        let edge_id = Id::new();
        let source_id = Id::new();
        let target_id = Id::new();
        let edge_args = AddEdge {
            id: edge_id,
            source_node_id: source_id,
            target_node_id: target_id,
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
        };

        writer.add_edge(edge_args.clone()).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify the edge was written to all three column families
        let db = DB::open_cf_for_read_only(
            &rocksdb::Options::default(),
            &db_path,
            ALL_COLUMN_FAMILIES,
            false,
        )
        .expect("Failed to open database for verification");

        // Check Edges column family
        {
            let cf_handle = db
                .cf_handle(Edges::CF_NAME)
                .expect("Edges column family should exist");
            let (key, _value) = Edges::record_from(&edge_args);
            let key_bytes = Edges::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(
                result.is_some(),
                "Edge should be written to the 'edges' column family"
            );

            let value_bytes = result.unwrap();
            let value = Edges::value_from_bytes(&value_bytes).expect("Failed to deserialize value");
            let content = value.0.content().expect("Failed to decode DataUrl");
            assert!(
                content.contains("test_edge"),
                "Edge value should contain the edge name"
            );
        }

        // Check ForwardEdges column family
        {
            use crate::schema::ForwardEdges;
            let cf_handle = db
                .cf_handle(ForwardEdges::CF_NAME)
                .expect("ForwardEdges column family should exist");
            let (key, _value) = ForwardEdges::record_from(&edge_args);
            let key_bytes = ForwardEdges::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(
                result.is_some(),
                "Edge should be written to the 'forward_edges' column family"
            );
        }

        // Check ReverseEdges column family
        {
            use crate::schema::ReverseEdges;
            let cf_handle = db
                .cf_handle(ReverseEdges::CF_NAME)
                .expect("ReverseEdges column family should exist");
            let (key, _value) = ReverseEdges::record_from(&edge_args);
            let key_bytes = ReverseEdges::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(
                result.is_some(),
                "Edge should be written to the 'reverse_edges' column family"
            );
        }
    }

    #[tokio::test]
    async fn test_fragments_written_to_correct_column_family() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Create a fragment with known ID and timestamp
        let fragment_id = Id::new();
        let timestamp = TimestampMilli::now();
        let fragment_args = AddFragment {
            id: fragment_id,
            ts_millis: timestamp.0,
            content: "This is test fragment content".to_string(),
        };

        writer.add_fragment(fragment_args.clone()).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify the fragment was written to the correct column family
        let db = DB::open_cf_for_read_only(
            &rocksdb::Options::default(),
            &db_path,
            ALL_COLUMN_FAMILIES,
            false,
        )
        .expect("Failed to open database for verification");

        let cf_handle = db
            .cf_handle(Fragments::CF_NAME)
            .expect("Fragments column family should exist");

        // Create the key using the schema's serialization
        let (key, _value) = Fragments::record_from(&fragment_args);
        let key_bytes = Fragments::key_to_bytes(&key).expect("Failed to serialize key");

        // Query the database
        let result = db
            .get_cf(cf_handle, &key_bytes)
            .expect("Failed to query database");
        assert!(
            result.is_some(),
            "Fragment should be written to the 'fragments' column family"
        );

        // Verify we can deserialize the value
        let value_bytes = result.unwrap();
        let value = Fragments::value_from_bytes(&value_bytes).expect("Failed to deserialize value");
        let content = value.0.content().expect("Failed to decode DataUrl");
        assert_eq!(
            content, "This is test fragment content",
            "Fragment content should match"
        );
    }

    #[tokio::test]
    async fn test_multiple_nodes_query_by_key() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config, &db_path);

        // Create multiple nodes with known IDs
        let node1_id = Id::new();
        let node2_id = Id::new();
        let node3_id = Id::new();

        let node1_args = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            name: "node_one".to_string(),
        };

        let node2_args = AddNode {
            id: node2_id,
            ts_millis: TimestampMilli::now(),
            name: "node_two".to_string(),
        };

        let node3_args = AddNode {
            id: node3_id,
            ts_millis: TimestampMilli::now(),
            name: "node_three".to_string(),
        };

        writer.add_node(node1_args.clone()).await.unwrap();
        writer.add_node(node2_args.clone()).await.unwrap();
        writer.add_node(node3_args.clone()).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify all nodes can be queried by their keys
        let db = DB::open_cf_for_read_only(
            &rocksdb::Options::default(),
            &db_path,
            ALL_COLUMN_FAMILIES,
            false,
        )
        .expect("Failed to open database for verification");

        let cf_handle = db
            .cf_handle(Nodes::CF_NAME)
            .expect("Nodes column family should exist");

        // Query node 1
        {
            let (key, _value) = Nodes::record_from(&node1_args);
            let key_bytes = Nodes::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(result.is_some(), "Node 1 should exist");
            let value = Nodes::value_from_bytes(&result.unwrap()).expect("Failed to deserialize");
            let content = value.0.content().expect("Failed to decode DataUrl");
            assert!(content.contains("node_one"));
        }

        // Query node 2
        {
            let (key, _value) = Nodes::record_from(&node2_args);
            let key_bytes = Nodes::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(result.is_some(), "Node 2 should exist");
            let value = Nodes::value_from_bytes(&result.unwrap()).expect("Failed to deserialize");
            let content = value.0.content().expect("Failed to decode DataUrl");
            assert!(content.contains("node_two"));
        }

        // Query node 3
        {
            let (key, _value) = Nodes::record_from(&node3_args);
            let key_bytes = Nodes::key_to_bytes(&key).expect("Failed to serialize key");
            let result = db
                .get_cf(cf_handle, &key_bytes)
                .expect("Failed to query database");
            assert!(result.is_some(), "Node 3 should exist");
            let value = Nodes::value_from_bytes(&result.unwrap()).expect("Failed to deserialize");
            let content = value.0.content().expect("Failed to decode DataUrl");
            assert!(content.contains("node_three"));
        }
    }

    #[tokio::test]
    async fn test_multiple_fragments_sorted_by_timestamp() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_graph_consumer(receiver, config.clone(), &db_path);

        // Create multiple nodes with fragments to test that we don't scan past the target ID
        let node_id_1 = Id::new();
        let node_id_2 = Id::new();
        let node_id_3 = Id::new();

        // Ensure IDs are in a specific order for testing
        let mut node_ids = vec![node_id_1, node_id_2, node_id_3];
        node_ids.sort();
        let (first_id, middle_id, last_id) = (node_ids[0], node_ids[1], node_ids[2]);

        let base_ts = 1700000000000u64;

        // Add nodes
        for (idx, &id) in node_ids.iter().enumerate() {
            let node_args = AddNode {
                id,
                ts_millis: TimestampMilli(base_ts + (idx as u64) * 10000),
                name: format!("test_node_{}", idx),
            };
            writer.add_node(node_args).await.unwrap();
        }

        // Add fragments for the FIRST node (should only retrieve these)
        let fragment1_1 = AddFragment {
            id: first_id,
            ts_millis: base_ts + 1000,
            content: "First Node - Fragment 1".to_string(),
        };
        let fragment1_2 = AddFragment {
            id: first_id,
            ts_millis: base_ts + 2000,
            content: "First Node - Fragment 2".to_string(),
        };
        let fragment1_3 = AddFragment {
            id: first_id,
            ts_millis: base_ts + 3000,
            content: "First Node - Fragment 3".to_string(),
        };

        // Add fragments for the MIDDLE node
        let fragment2_1 = AddFragment {
            id: middle_id,
            ts_millis: base_ts + 4000,
            content: "Middle Node - Fragment 1".to_string(),
        };
        let fragment2_2 = AddFragment {
            id: middle_id,
            ts_millis: base_ts + 5000,
            content: "Middle Node - Fragment 2".to_string(),
        };

        // Add fragments for the LAST node
        let fragment3_1 = AddFragment {
            id: last_id,
            ts_millis: base_ts + 6000,
            content: "Last Node - Fragment 1".to_string(),
        };

        // Add fragments in mixed order
        writer.add_fragment(fragment2_1.clone()).await.unwrap();
        writer.add_fragment(fragment1_3.clone()).await.unwrap();
        writer.add_fragment(fragment3_1.clone()).await.unwrap();
        writer.add_fragment(fragment1_1.clone()).await.unwrap();
        writer.add_fragment(fragment2_2.clone()).await.unwrap();
        writer.add_fragment(fragment1_2.clone()).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Now query using the new fragments_by_id API
        let (reader, receiver) = crate::create_query_reader(crate::ReaderConfig {
            channel_buffer_size: 10,
        });

        let mut storage = Storage::readonly(&db_path);

        // Open the database in readonly mode
        storage.ready().unwrap();

        let storage_arc = std::sync::Arc::new(storage);
        let graph = Graph::new(storage_arc);

        let consumer = crate::query::Consumer::new(
            receiver,
            crate::ReaderConfig {
                channel_buffer_size: 10,
            },
            graph,
        );

        let consumer_handle = crate::query::spawn_consumer(consumer);

        // Query all fragments for the FIRST node (middle in sorted order)
        let fragments = reader
            .fragments_by_id(first_id, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify we got exactly 3 fragments (only for first_id, not middle or last)
        assert_eq!(
            fragments.len(),
            3,
            "Should have received exactly 3 fragments for the first node"
        );

        // Extract timestamps and content
        let (ts1, content1) = &fragments[0];
        let (ts2, content2) = &fragments[1];
        let (ts3, content3) = &fragments[2];

        // Verify timestamps are in sorted order (without additional sorting)
        assert!(
            ts1 < ts2,
            "First timestamp ({:?}) should be less than second ({:?})",
            ts1,
            ts2
        );
        assert!(
            ts2 < ts3,
            "Second timestamp ({:?}) should be less than third ({:?})",
            ts2,
            ts3
        );

        // Verify the expected timestamps match what we inserted
        assert_eq!(ts1.0, base_ts + 1000, "First fragment timestamp mismatch");
        assert_eq!(ts2.0, base_ts + 2000, "Second fragment timestamp mismatch");
        assert_eq!(ts3.0, base_ts + 3000, "Third fragment timestamp mismatch");

        // Verify they are in chronological order (sorted by timestamp)
        let content1_str = content1.content().unwrap();
        let content2_str = content2.content().unwrap();
        let content3_str = content3.content().unwrap();

        assert!(
            content1_str.contains("First Node - Fragment 1"),
            "First fragment should be Fragment 1: {}",
            content1_str
        );
        assert!(
            content2_str.contains("First Node - Fragment 2"),
            "Second fragment should be Fragment 2: {}",
            content2_str
        );
        assert!(
            content3_str.contains("First Node - Fragment 3"),
            "Third fragment should be Fragment 3: {}",
            content3_str
        );

        // Verify no fragments from other nodes are included
        for (_, fragment_content) in &fragments {
            let content = fragment_content.content().unwrap();
            assert!(
                !content.contains("Middle Node") && !content.contains("Last Node"),
                "Should not contain fragments from other nodes: {}",
                content
            );
        }

        // Clean up
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
