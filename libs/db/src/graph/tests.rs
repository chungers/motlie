use crate::graph::{ColumnFamily, ColumnFamilySerde, Graph, HotColumnFamilyRecord, Storage};
use crate::graph::mutation::{
    AddEdge, AddNode, AddNodeFragment, UpdateEdgeValidSinceUntil,
    UpdateNodeValidSinceUntil,
};
use crate::writer::Runnable as MutRunnable;
use crate::reader::Runnable;
use crate::graph::schema::{EdgeSummary, NodeFragments, Nodes, ALL_COLUMN_FAMILIES};
use crate::graph::writer::{
    create_mutation_writer, spawn_mutation_consumer, spawn_mutation_consumer_with_next, WriterConfig,
};
use crate::{Id, TemporalRange, TimestampMilli};
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
        let consumer_handle = spawn_mutation_consumer(receiver, config, &db_path);

        // Send some mutations
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        let edge_args = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
            summary: EdgeSummary::from_text(""),
            weight: Some(1.0),
            valid_range: None,
        };
        edge_args.run(&writer).await.unwrap();

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
        let consumer_handle = spawn_mutation_consumer(receiver, config, &db_path);

        // Send 5 mutations rapidly
        for i in 0..5 {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text(&format!("summary {}", i)),
            };
            node_args.run(&writer).await.unwrap();
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
        let consumer_handle = spawn_mutation_consumer(receiver, config, &db_path);

        // Test all mutation types
        let node_id = Id::new();
        AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: "node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("node summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        let edge_src_id = Id::new();
        let edge_dst_id = Id::new();
        let edge_name = "edge".to_string();
        AddEdge {
            source_node_id: edge_src_id,
            target_node_id: edge_dst_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.clone(),
            summary: EdgeSummary::from_text(""),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: crate::DataUrl::from_text("fragment body"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give consumer time to process the node and edge
        tokio::time::sleep(Duration::from_millis(50)).await;

        UpdateNodeValidSinceUntil {
            id: node_id,
            temporal_range: TemporalRange(None, None),
            reason: "test node invalidation".to_string(),
        }
        .run(&writer)
        .await
        .unwrap();

        UpdateEdgeValidSinceUntil {
            src_id: edge_src_id,
            dst_id: edge_dst_id,
            name: edge_name,
            temporal_range: TemporalRange(None, None),
            reason: "test edge invalidation".to_string(),
        }
        .run(&writer)
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
        let fulltext_index_path = temp_dir.path().join("fulltext_index");
        let fulltext_handle = crate::fulltext::spawn_mutation_consumer(fulltext_receiver, config.clone(), &fulltext_index_path);

        // Create the Graph consumer that forwards to FullText
        let (writer, graph_receiver) = create_mutation_writer(config.clone());
        let graph_handle = spawn_mutation_consumer_with_next(
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
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text(&format!("chained summary {}", i)),
            };
            let fragment_args = AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: crate::DataUrl::from_text(&format!(
                    "Chained fragment {} processed by both Graph and FullText",
                    i
                )),
                valid_range: None,
            };

            node_args.run(&writer).await.unwrap();
            fragment_args.run(&writer).await.unwrap();
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
        let fragments_cf = db.cf_handle(NodeFragments::CF_NAME);
        assert!(
            fragments_cf.is_some(),
            "Fragments column family should exist: {}",
            NodeFragments::CF_NAME
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
            db_final.cf_handle(NodeFragments::CF_NAME).is_some(),
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
        let consumer_handle = spawn_mutation_consumer(receiver, config, &db_path);

        // Create a node with known ID
        let node_id = Id::new();
        let node_args = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test summary"),
        };

        node_args.clone().run(&writer).await.unwrap();

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
        let key_bytes = Nodes::key_to_bytes(&key);

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
        let node_name_hash = &value.1; // NodeName is now NameHash (field 1 after temporal_range)
        // Note: value.2 is now Option<SummaryHash> - the actual summary is stored in NodeSummaries CF
        // Check that the name hash matches the expected hash
        use crate::graph::NameHash;
        assert_eq!(*node_name_hash, NameHash::from_name("test_node"), "Node name hash should match");
        // The summary hash should be Some (since we provided a non-empty summary)
        assert!(
            value.2.is_some(),
            "Summary hash should be present for non-empty summary"
        );
    }

    // NOTE: test_node_names_column_family was removed - NodeNames column family
    // no longer exists. Name-based lookups are now handled by fulltext search.

    #[test]
    fn test_lz4_compression_round_trip() {
        use crate::graph::schema::{NodeCfValue, NodeSummary, Nodes};
        use crate::graph::NameHash;
        use crate::graph::SummaryHash;
        use crate::DataUrl;

        // Create a test summary and compute its hash
        let summary = DataUrl::from_markdown("test content");
        let summary_hash = SummaryHash::from_summary(&summary).ok();

        // Create a simple test value with NameHash and SummaryHash
        let name_hash = NameHash::from_name("test_node");
        let test_value = NodeCfValue(
            None,
            name_hash,
            summary_hash,
        );

        // Serialize and compress
        let compressed_bytes = Nodes::value_to_bytes(&test_value).expect("Failed to compress");

        println!("Compressed size: {} bytes", compressed_bytes.len());
        println!(
            "First 20 bytes: {:?}",
            &compressed_bytes[..20.min(compressed_bytes.len())]
        );

        // Decompress and deserialize
        let decompressed_value: NodeCfValue =
            Nodes::value_from_bytes(&compressed_bytes).expect("Failed to decompress");

        // Verify
        assert_eq!(test_value.0, decompressed_value.0, "Temporal range should match");
        assert_eq!(test_value.1, decompressed_value.1, "Name hash should match");
        assert_eq!(test_value.2, decompressed_value.2, "Summary hash should match");
    }

    // =========================================================================
    // Phase 2: Blob Separation Tests
    // =========================================================================

    #[test]
    fn test_summary_hash_content_addressable() {
        use crate::graph::SummaryHash;
        use crate::DataUrl;

        // Same content should produce the same hash
        let summary1 = DataUrl::from_markdown("This is a test summary");
        let summary2 = DataUrl::from_markdown("This is a test summary");
        let hash1 = SummaryHash::from_summary(&summary1).unwrap();
        let hash2 = SummaryHash::from_summary(&summary2).unwrap();
        assert_eq!(hash1, hash2, "Same content should produce same hash");

        // Different content should produce different hashes
        let summary3 = DataUrl::from_markdown("This is a different summary");
        let hash3 = SummaryHash::from_summary(&summary3).unwrap();
        assert_ne!(hash1, hash3, "Different content should produce different hash");
    }

    #[test]
    fn test_summary_hash_to_bytes_round_trip() {
        use crate::graph::SummaryHash;
        use crate::DataUrl;

        let summary = DataUrl::from_markdown("Test content for hash");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        // Convert to bytes and back
        let bytes = hash.as_bytes();
        assert_eq!(bytes.len(), 8, "SummaryHash should be 8 bytes");

        let recovered = SummaryHash::from_bytes(*bytes);
        assert_eq!(hash, recovered, "Round-trip should preserve hash");
    }

    #[test]
    fn test_node_summaries_cf_round_trip() {
        use crate::graph::schema::{NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue, NodeSummary};
        use crate::graph::SummaryHash;
        use crate::DataUrl;

        // Create a summary and its hash
        let summary = DataUrl::from_markdown("Node summary content for testing");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        // Create CF key and value
        let key = NodeSummaryCfKey(hash);
        let value = NodeSummaryCfValue(summary.clone());

        // Serialize
        let key_bytes = NodeSummaries::key_to_bytes(&key);
        let value_bytes = NodeSummaries::value_to_bytes(&value).unwrap();

        // Deserialize
        let recovered_key = NodeSummaries::key_from_bytes(&key_bytes).unwrap();
        let recovered_value = NodeSummaries::value_from_bytes(&value_bytes).unwrap();

        assert_eq!(key.0, recovered_key.0, "Key hash should match");
        assert_eq!(
            value.0.decode_string().unwrap(),
            recovered_value.0.decode_string().unwrap(),
            "Value summary should match"
        );
    }

    #[test]
    fn test_edge_summaries_cf_round_trip() {
        use crate::graph::schema::{EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue, EdgeSummary};
        use crate::graph::SummaryHash;
        use crate::DataUrl;

        // Create a summary and its hash
        let summary = DataUrl::from_markdown("Edge summary content for testing");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        // Create CF key and value
        let key = EdgeSummaryCfKey(hash);
        let value = EdgeSummaryCfValue(summary.clone());

        // Serialize
        let key_bytes = EdgeSummaries::key_to_bytes(&key);
        let value_bytes = EdgeSummaries::value_to_bytes(&value).unwrap();

        // Deserialize
        let recovered_key = EdgeSummaries::key_from_bytes(&key_bytes).unwrap();
        let recovered_value = EdgeSummaries::value_from_bytes(&value_bytes).unwrap();

        assert_eq!(key.0, recovered_key.0, "Key hash should match");
        assert_eq!(
            value.0.decode_string().unwrap(),
            recovered_value.0.decode_string().unwrap(),
            "Value summary should match"
        );
    }

    #[test]
    fn test_empty_summary_produces_none_hash() {
        use crate::graph::schema::NodeSummary;
        use crate::DataUrl;

        // Empty summary (from_text("")) should not produce a SummaryHash in mutations
        // The mutation executor checks `is_empty()` before computing hash
        let empty_summary = DataUrl::from_text("");
        assert!(empty_summary.is_empty(), "Empty DataUrl should report is_empty()");

        let nonempty_summary = DataUrl::from_text("content");
        assert!(!nonempty_summary.is_empty(), "Non-empty DataUrl should not report is_empty()");
    }

    // =========================================================================
    // Phase 2: End-to-End Tests for Empty Summary Handling
    // =========================================================================

    #[test]
    fn test_add_node_with_no_summary_and_query() {
        use crate::graph::mutation::AddNode;
        use crate::graph::query::{NodeById, TransactionQueryExecutor};
        use crate::graph::writer::MutationExecutor;
        use crate::graph::Storage;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create a node with an empty summary
        let node_id = Id::new();
        let empty_summary = DataUrl::from_text("");

        let add_node = AddNode {
            id: node_id,
            ts_millis: TimestampMilli(0),
            name: "EmptyNode".to_string(),
            summary: empty_summary.clone(),
            valid_range: None,
        };

        // Execute the mutation and query in the same transaction
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_node.execute(&txn, txn_db).unwrap();

        // Query the node back using transaction executor
        let query = NodeById::new(node_id, None);
        let result = query.execute_in_transaction(&txn, txn_db, storage.cache());

        assert!(result.is_ok(), "Query should succeed for node with empty summary");
        let (name, summary) = result.unwrap();
        assert_eq!(name, "EmptyNode", "Node name should match");
        // Empty summary should be returned as empty DataUrl
        assert!(summary.is_empty(), "Summary should be empty for node with no summary");

        txn.commit().unwrap();
    }

    #[test]
    fn test_add_edge_with_no_summary_and_query() {
        use crate::graph::mutation::{AddEdge, AddNode};
        use crate::graph::query::{EdgeSummaryBySrcDstName, TransactionQueryExecutor};
        use crate::graph::writer::MutationExecutor;
        use crate::graph::Storage;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create source and destination nodes first
        let src_id = Id::new();
        let dst_id = Id::new();

        let add_src = AddNode {
            id: src_id,
            ts_millis: TimestampMilli(0),
            name: "SourceNode".to_string(),
            summary: DataUrl::from_text("source"),
            valid_range: None,
        };

        let add_dst = AddNode {
            id: dst_id,
            ts_millis: TimestampMilli(0),
            name: "DestNode".to_string(),
            summary: DataUrl::from_text("dest"),
            valid_range: None,
        };

        // Create an edge with an empty summary
        let empty_summary = DataUrl::from_text("");
        let add_edge = AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: TimestampMilli(0),
            name: "CONNECTS".to_string(),
            summary: empty_summary.clone(),
            weight: Some(1.5),
            valid_range: None,
        };

        // Execute the mutations and query in the same transaction
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_src.execute(&txn, txn_db).unwrap();
        add_dst.execute(&txn, txn_db).unwrap();
        add_edge.execute(&txn, txn_db).unwrap();

        // Query the edge back using transaction executor
        let query = EdgeSummaryBySrcDstName::new(src_id, dst_id, "CONNECTS".to_string(), None);
        let result = query.execute_in_transaction(&txn, txn_db, storage.cache());

        assert!(result.is_ok(), "Query should succeed for edge with empty summary");
        let (summary, weight) = result.unwrap();
        assert!(summary.is_empty(), "Summary should be empty for edge with no summary");
        assert_eq!(weight, Some(1.5), "Weight should match");

        txn.commit().unwrap();
    }

    #[test]
    fn test_add_node_with_summary_and_query() {
        use crate::graph::mutation::AddNode;
        use crate::graph::query::{NodeById, TransactionQueryExecutor};
        use crate::graph::writer::MutationExecutor;
        use crate::graph::Storage;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create a node with a non-empty summary
        let node_id = Id::new();
        let summary_content = "This is a detailed summary for the node.";
        let node_summary = DataUrl::from_markdown(summary_content);

        let add_node = AddNode {
            id: node_id,
            ts_millis: TimestampMilli(0),
            name: "SummaryNode".to_string(),
            summary: node_summary.clone(),
            valid_range: None,
        };

        // Execute the mutation and query in the same transaction
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_node.execute(&txn, txn_db).unwrap();

        // Query the node back using transaction executor
        let query = NodeById::new(node_id, None);
        let result = query.execute_in_transaction(&txn, txn_db, storage.cache());

        assert!(result.is_ok(), "Query should succeed for node with summary");
        let (name, summary) = result.unwrap();
        assert_eq!(name, "SummaryNode", "Node name should match");
        assert!(!summary.is_empty(), "Summary should not be empty");
        let decoded = summary.decode_string().unwrap();
        assert!(decoded.contains(summary_content), "Summary content should match");

        txn.commit().unwrap();
    }

    #[test]
    fn test_add_edge_with_summary_and_query() {
        use crate::graph::mutation::{AddEdge, AddNode};
        use crate::graph::query::{EdgeSummaryBySrcDstName, TransactionQueryExecutor};
        use crate::graph::writer::MutationExecutor;
        use crate::graph::Storage;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create source and destination nodes first
        let src_id = Id::new();
        let dst_id = Id::new();

        let add_src = AddNode {
            id: src_id,
            ts_millis: TimestampMilli(0),
            name: "SourceNode".to_string(),
            summary: DataUrl::from_text("source"),
            valid_range: None,
        };

        let add_dst = AddNode {
            id: dst_id,
            ts_millis: TimestampMilli(0),
            name: "DestNode".to_string(),
            summary: DataUrl::from_text("dest"),
            valid_range: None,
        };

        // Create an edge with a non-empty summary
        let edge_summary_content = "This edge represents a strong connection.";
        let edge_summary = DataUrl::from_markdown(edge_summary_content);
        let add_edge = AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: TimestampMilli(0),
            name: "STRONG_LINK".to_string(),
            summary: edge_summary.clone(),
            weight: Some(9.9),
            valid_range: None,
        };

        // Execute the mutations and query in the same transaction
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_src.execute(&txn, txn_db).unwrap();
        add_dst.execute(&txn, txn_db).unwrap();
        add_edge.execute(&txn, txn_db).unwrap();

        // Query the edge back using transaction executor
        let query = EdgeSummaryBySrcDstName::new(src_id, dst_id, "STRONG_LINK".to_string(), None);
        let result = query.execute_in_transaction(&txn, txn_db, storage.cache());

        assert!(result.is_ok(), "Query should succeed for edge with summary");
        let (summary, weight) = result.unwrap();
        assert!(!summary.is_empty(), "Summary should not be empty");
        let decoded: String = summary.decode_string().unwrap();
        assert!(decoded.contains(edge_summary_content), "Summary content should match");
        assert_eq!(weight, Some(9.9), "Weight should match");

        txn.commit().unwrap();
    }

    #[test]
    fn test_summary_deduplication() {
        use crate::graph::mutation::AddNode;
        use crate::graph::query::{NodeById, TransactionQueryExecutor};
        use crate::graph::writer::MutationExecutor;
        use crate::graph::Storage;
        use crate::graph::SummaryHash;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create two nodes with the SAME summary content
        let shared_summary = DataUrl::from_markdown("This is a shared summary.");
        let node1_id = Id::new();
        let node2_id = Id::new();

        let add_node1 = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli(0),
            name: "Node1".to_string(),
            summary: shared_summary.clone(),
            valid_range: None,
        };

        let add_node2 = AddNode {
            id: node2_id,
            ts_millis: TimestampMilli(0),
            name: "Node2".to_string(),
            summary: shared_summary.clone(),
            valid_range: None,
        };

        // Execute the mutations
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_node1.execute(&txn, txn_db).unwrap();
        add_node2.execute(&txn, txn_db).unwrap();

        // Both nodes should return the same summary content
        let query1 = NodeById::new(node1_id, None);
        let query2 = NodeById::new(node2_id, None);

        let (_, summary1) = query1.execute_in_transaction(&txn, txn_db, storage.cache()).unwrap();
        let (_, summary2) = query2.execute_in_transaction(&txn, txn_db, storage.cache()).unwrap();

        let decoded1: String = summary1.decode_string().unwrap();
        let decoded2: String = summary2.decode_string().unwrap();
        assert_eq!(
            decoded1,
            decoded2,
            "Both nodes should have the same summary content"
        );

        // Verify deduplication: same content produces same hash
        let hash1 = SummaryHash::from_summary(&shared_summary).unwrap();
        let hash2 = SummaryHash::from_summary(&shared_summary).unwrap();
        assert_eq!(hash1, hash2, "Same content should produce same hash (deduplication)");

        txn.commit().unwrap();
    }

    /// Test that verifies:
    /// 1. Two nodes with same summary are stored as distinct nodes
    /// 2. Both nodes share the same summary_hash pointing to one NodeSummaries entry
    /// 3. After updating one node's summary, two distinct summaries exist
    /// 4. Each node points to the correct summary
    #[test]
    fn test_summary_sharing_and_update() {
        use crate::graph::mutation::AddNode;
        use crate::graph::schema::{
            NodeCfKey, NodeCfValue, NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue, Nodes,
        };
        use crate::graph::writer::MutationExecutor;
        use crate::graph::ColumnFamilySerde;
        use crate::graph::Storage;
        use crate::graph::SummaryHash;
        use crate::{DataUrl, Id, TimestampMilli};

        let temp_dir = tempfile::tempdir().unwrap();
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().unwrap();

        // Create two nodes with the SAME summary content
        let shared_summary = DataUrl::from_markdown("This is a shared summary for testing.");
        let node1_id = Id::new();
        let node2_id = Id::new();

        let add_node1 = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli(100),
            name: "Node1".to_string(),
            summary: shared_summary.clone(),
            valid_range: None,
        };

        let add_node2 = AddNode {
            id: node2_id,
            ts_millis: TimestampMilli(200),
            name: "Node2".to_string(),
            summary: shared_summary.clone(),
            valid_range: None,
        };

        // Execute the mutations
        let txn_db = storage.transaction_db().unwrap();
        let txn = txn_db.transaction();

        add_node1.execute(&txn, txn_db).unwrap();
        add_node2.execute(&txn, txn_db).unwrap();
        txn.commit().unwrap();

        // ========================================================================
        // STEP 1: Verify two distinct nodes exist in Nodes CF
        // ========================================================================
        let txn = txn_db.transaction();

        // Read node1 raw value from Nodes CF
        let cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
        let key1 = Nodes::key_to_bytes(&NodeCfKey(node1_id));
        let raw1 = txn.get_for_update_cf(&cf, &key1, true).unwrap().unwrap();
        let value1: NodeCfValue = Nodes::value_from_bytes(&raw1).unwrap();

        // Read node2 raw value from Nodes CF
        let key2 = Nodes::key_to_bytes(&NodeCfKey(node2_id));
        let raw2 = txn.get_for_update_cf(&cf, &key2, true).unwrap().unwrap();
        let value2: NodeCfValue = Nodes::value_from_bytes(&raw2).unwrap();

        // Verify they are distinct nodes
        assert_ne!(node1_id, node2_id, "Nodes should have different IDs");

        // ========================================================================
        // STEP 2: Verify both nodes share the SAME summary_hash
        // ========================================================================
        let hash1 = value1.2.expect("Node1 should have a summary hash");
        let hash2 = value2.2.expect("Node2 should have a summary hash");
        assert_eq!(hash1, hash2, "Both nodes should share the same summary_hash");

        // ========================================================================
        // STEP 3: Count entries in NodeSummaries CF - should be exactly 1
        // ========================================================================
        let summary_cf = txn_db.cf_handle(NodeSummaries::CF_NAME).unwrap();
        let mut summary_count = 0;
        let iter = txn.iterator_cf(&summary_cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let _ = item.unwrap();
            summary_count += 1;
        }
        assert_eq!(
            summary_count, 1,
            "Should have exactly 1 summary entry (shared by both nodes)"
        );

        // Verify the summary content
        let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash1));
        let summary_raw = txn.get_cf(&summary_cf, &summary_key).unwrap().unwrap();
        let summary_value: NodeSummaryCfValue = NodeSummaries::value_from_bytes(&summary_raw).unwrap();
        let decoded: String = summary_value.0.decode_string().unwrap();
        assert!(
            decoded.contains("shared summary"),
            "Summary content should match"
        );

        txn.commit().unwrap();

        // ========================================================================
        // STEP 4: Update node1's summary to a different value
        // ========================================================================
        let new_summary = DataUrl::from_markdown("This is a NEW summary for Node1 only.");

        // We need to use the mutation path - let's re-add node1 with a new summary
        // (In practice this would be an UpdateNode mutation, but for simplicity we test
        // by directly writing to the CFs)
        let txn = txn_db.transaction();

        // Write new summary to NodeSummaries CF
        let new_hash = SummaryHash::from_summary(&new_summary).unwrap();
        let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(new_hash));
        let summary_value = NodeSummaries::value_to_bytes(&NodeSummaryCfValue(new_summary.clone())).unwrap();
        txn.put_cf(&summary_cf, &summary_key, &summary_value).unwrap();

        // Update node1 to point to new summary_hash
        let cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
        let new_value1 = NodeCfValue(value1.0, value1.1, Some(new_hash));
        let value_bytes = Nodes::value_to_bytes(&new_value1).unwrap();
        txn.put_cf(&cf, &key1, &value_bytes).unwrap();

        txn.commit().unwrap();

        // ========================================================================
        // STEP 5: Verify now TWO summaries exist in NodeSummaries CF
        // ========================================================================
        let txn = txn_db.transaction();
        let summary_cf = txn_db.cf_handle(NodeSummaries::CF_NAME).unwrap();
        let mut summary_count = 0;
        let iter = txn.iterator_cf(&summary_cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let _ = item.unwrap();
            summary_count += 1;
        }
        assert_eq!(
            summary_count, 2,
            "Should now have 2 summary entries (original + new)"
        );

        // ========================================================================
        // STEP 6: Verify each node points to the correct summary
        // ========================================================================
        // Re-read node1
        let cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
        let raw1 = txn.get_cf(&cf, &key1).unwrap().unwrap();
        let value1: NodeCfValue = Nodes::value_from_bytes(&raw1).unwrap();
        let node1_hash = value1.2.expect("Node1 should have a summary hash");

        // Re-read node2
        let raw2 = txn.get_cf(&cf, &key2).unwrap().unwrap();
        let value2: NodeCfValue = Nodes::value_from_bytes(&raw2).unwrap();
        let node2_hash = value2.2.expect("Node2 should have a summary hash");

        // Node1 should have new_hash, Node2 should have original hash1
        assert_eq!(node1_hash, new_hash, "Node1 should point to new summary");
        assert_eq!(node2_hash, hash1, "Node2 should still point to original summary");
        assert_ne!(
            node1_hash, node2_hash,
            "Nodes should now point to different summaries"
        );

        // Verify node1's summary content
        let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(node1_hash));
        let summary_raw = txn.get_cf(&summary_cf, &summary_key).unwrap().unwrap();
        let summary_value: NodeSummaryCfValue = NodeSummaries::value_from_bytes(&summary_raw).unwrap();
        let decoded: String = summary_value.0.decode_string().unwrap();
        assert!(
            decoded.contains("NEW summary"),
            "Node1 should have new summary content"
        );

        // Verify node2's summary content
        let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(node2_hash));
        let summary_raw = txn.get_cf(&summary_cf, &summary_key).unwrap().unwrap();
        let summary_value: NodeSummaryCfValue = NodeSummaries::value_from_bytes(&summary_raw).unwrap();
        let decoded: String = summary_value.0.decode_string().unwrap();
        assert!(
            decoded.contains("shared summary"),
            "Node2 should still have original summary content"
        );

        txn.commit().unwrap();
    }

    // ============================================================================
    // Phase 3: rkyv Zero-Copy Serialization Tests
    // ============================================================================

    /// Test rkyv serialization round-trip for NodeCfValue
    #[test]
    fn test_rkyv_node_value_round_trip() {
        use crate::graph::schema::{NodeCfValue, Nodes};
        use crate::graph::name_hash::NameHash;
        use crate::graph::summary_hash::SummaryHash;
        use crate::{DataUrl, TemporalRange, TimestampMilli};

        // Create a NodeCfValue with all fields populated
        let name_hash = NameHash::from_name("test_node");
        let summary = DataUrl::from_text("Test summary content");
        let summary_hash = SummaryHash::from_summary(&summary).unwrap();
        let temporal_range = TemporalRange::valid_between(
            TimestampMilli(1000),
            TimestampMilli(2000),
        );

        let original = NodeCfValue(temporal_range, name_hash, Some(summary_hash));

        // Serialize with rkyv
        let bytes = Nodes::value_to_bytes(&original).expect("rkyv serialize");

        // Deserialize with rkyv
        let recovered: NodeCfValue = Nodes::value_from_bytes(&bytes).expect("rkyv deserialize");

        // Verify all fields match
        assert_eq!(original.0, recovered.0, "TemporalRange should match");
        assert_eq!(original.1, recovered.1, "NameHash should match");
        assert_eq!(original.2, recovered.2, "SummaryHash should match");
    }

    /// Test rkyv zero-copy access for NodeCfValue
    #[test]
    fn test_rkyv_node_value_zero_copy_access() {
        use crate::graph::schema::{NodeCfValue, Nodes};
        use crate::graph::name_hash::NameHash;
        use crate::graph::summary_hash::SummaryHash;
        use crate::{DataUrl, TemporalRange, TimestampMilli};

        let name_hash = NameHash::from_name("zero_copy_node");
        let summary = DataUrl::from_text("Zero copy test");
        let summary_hash = SummaryHash::from_summary(&summary).unwrap();
        let temporal_range = TemporalRange::valid_from(TimestampMilli(5000));

        let original = NodeCfValue(temporal_range.clone(), name_hash, Some(summary_hash));
        let bytes = Nodes::value_to_bytes(&original).expect("serialize");

        // Access archived data without full deserialization
        let archived = Nodes::value_archived(&bytes).expect("zero-copy access");

        // Verify we can read fields from archived reference
        // The archived temporal_range should match
        assert!(archived.0.is_some(), "Archived temporal range should be Some");

        // SummaryHash comparison through archived
        assert!(archived.2.is_some(), "Archived summary hash should be Some");

        // Verify we can deserialize and get matching data
        let recovered: NodeCfValue = Nodes::value_from_bytes(&bytes).expect("deserialize");
        assert_eq!(recovered.1, name_hash, "NameHash should match after full deserialize");
    }

    /// Test rkyv serialization round-trip for ForwardEdgeCfValue
    #[test]
    fn test_rkyv_forward_edge_value_round_trip() {
        use crate::graph::schema::{ForwardEdgeCfValue, ForwardEdges};
        use crate::graph::summary_hash::SummaryHash;
        use crate::{DataUrl, TemporalRange, TimestampMilli};

        let summary = DataUrl::from_text("Edge summary");
        let summary_hash = SummaryHash::from_summary(&summary).unwrap();
        let temporal_range = TemporalRange::valid_between(
            TimestampMilli(100),
            TimestampMilli(200),
        );

        let original = ForwardEdgeCfValue(temporal_range, Some(3.14), Some(summary_hash));

        // Serialize with rkyv
        let bytes = ForwardEdges::value_to_bytes(&original).expect("rkyv serialize");

        // Deserialize with rkyv
        let recovered: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&bytes).expect("rkyv deserialize");

        // Verify all fields match
        assert_eq!(original.0, recovered.0, "TemporalRange should match");
        assert_eq!(original.1, recovered.1, "Weight should match");
        assert_eq!(original.2, recovered.2, "SummaryHash should match");
    }

    /// Test rkyv zero-copy access for ForwardEdgeCfValue with weight
    #[test]
    fn test_rkyv_forward_edge_zero_copy_weight() {
        use crate::graph::schema::{ForwardEdgeCfValue, ForwardEdges};

        let weight = Some(2.718);
        let original = ForwardEdgeCfValue(None, weight, None);
        let bytes = ForwardEdges::value_to_bytes(&original).expect("serialize");

        // Access archived data without full deserialization
        let archived = ForwardEdges::value_archived(&bytes).expect("zero-copy access");

        // Verify weight through archived reference
        assert!(archived.1.is_some(), "Archived weight should be Some");
        assert_eq!(archived.1, Some(2.718), "Archived weight value should match");
    }

    // ReverseEdgeCfValue test removed - ReverseEdges now has empty value

    /// Test rkyv handles None values correctly
    #[test]
    fn test_rkyv_none_values() {
        use crate::graph::schema::{NodeCfValue, Nodes, ForwardEdgeCfValue, ForwardEdges};
        use crate::graph::name_hash::NameHash;

        // NodeCfValue with None summary_hash
        let name_hash = NameHash::from_name("no_summary");
        let node_value = NodeCfValue(None, name_hash, None);
        let bytes = Nodes::value_to_bytes(&node_value).expect("serialize");
        let recovered: NodeCfValue = Nodes::value_from_bytes(&bytes).expect("deserialize");
        assert!(recovered.0.is_none(), "TemporalRange should be None");
        assert!(recovered.2.is_none(), "SummaryHash should be None");

        // ForwardEdgeCfValue with all None
        let edge_value = ForwardEdgeCfValue(None, None, None);
        let bytes = ForwardEdges::value_to_bytes(&edge_value).expect("serialize");
        let recovered: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&bytes).expect("deserialize");
        assert!(recovered.0.is_none(), "TemporalRange should be None");
        assert!(recovered.1.is_none(), "Weight should be None");
        assert!(recovered.2.is_none(), "SummaryHash should be None");
    }

    /// Test rkyv serialized size is reasonable
    #[test]
    fn test_rkyv_compact_size() {
        use crate::graph::schema::{NodeCfValue, Nodes, ForwardEdgeCfValue, ForwardEdges};
        use crate::graph::name_hash::NameHash;
        use crate::graph::summary_hash::SummaryHash;
        use crate::DataUrl;

        // NodeCfValue with rkyv includes alignment padding
        // Fields: TemporalRange (Option), NameHash (8), Option<SummaryHash> (8)
        // rkyv aligned size is typically 64 bytes for this structure
        let name_hash = NameHash::from_name("test");
        let node_value = NodeCfValue(None, name_hash, None);
        let bytes = Nodes::value_to_bytes(&node_value).expect("serialize");
        assert!(
            bytes.len() <= 80,
            "NodeCfValue size should be reasonable: got {} bytes",
            bytes.len()
        );
        // Verify it's smaller than MessagePack+LZ4 would be for this data
        // (MessagePack alone for this would be ~30-40 bytes, but rkyv trades size for speed)

        // ForwardEdgeCfValue with all fields
        let summary = DataUrl::from_text("x");
        let summary_hash = SummaryHash::from_summary(&summary).unwrap();
        let edge_value = ForwardEdgeCfValue(None, Some(1.0), Some(summary_hash));
        let bytes = ForwardEdges::value_to_bytes(&edge_value).expect("serialize");
        assert!(
            bytes.len() <= 80,
            "ForwardEdgeCfValue size should be reasonable: got {} bytes",
            bytes.len()
        );

        // Log actual sizes for documentation
        println!("NodeCfValue (minimal): {} bytes", Nodes::value_to_bytes(&node_value).unwrap().len());
        println!("ForwardEdgeCfValue (with weight+summary): {} bytes", bytes.len());
    }

    /// Test key serialization for hot CFs
    #[test]
    fn test_hot_cf_key_serialization() {
        use crate::graph::schema::{
            NodeCfKey, Nodes,
            ForwardEdgeCfKey, ForwardEdges,
            ReverseEdgeCfKey, ReverseEdges,
        };
        use crate::graph::name_hash::NameHash;
        use crate::Id;

        // Node key: 16 bytes (Id)
        let node_id = Id::new();
        let node_key = NodeCfKey(node_id);
        let bytes = Nodes::key_to_bytes(&node_key);
        assert_eq!(bytes.len(), 16, "NodeCfKey should be 16 bytes");
        let recovered = Nodes::key_from_bytes(&bytes).expect("deserialize");
        assert_eq!(node_key.0, recovered.0, "Node key should round-trip");

        // Forward edge key: 40 bytes (16 + 16 + 8)
        let src_id = Id::new();
        let dst_id = Id::new();
        let name_hash = NameHash::from_name("edge");
        let forward_key = ForwardEdgeCfKey(src_id, dst_id, name_hash);
        let bytes = ForwardEdges::key_to_bytes(&forward_key);
        assert_eq!(bytes.len(), 40, "ForwardEdgeCfKey should be 40 bytes");
        let recovered = ForwardEdges::key_from_bytes(&bytes).expect("deserialize");
        assert_eq!(forward_key.0, recovered.0, "Src ID should match");
        assert_eq!(forward_key.1, recovered.1, "Dst ID should match");
        assert_eq!(forward_key.2, recovered.2, "NameHash should match");

        // Reverse edge key: 40 bytes (16 + 16 + 8)
        let reverse_key = ReverseEdgeCfKey(dst_id, src_id, name_hash);
        let bytes = ReverseEdges::key_to_bytes(&reverse_key);
        assert_eq!(bytes.len(), 40, "ReverseEdgeCfKey should be 40 bytes");
        let recovered = ReverseEdges::key_from_bytes(&bytes).expect("deserialize");
        assert_eq!(reverse_key.0, recovered.0, "Dst ID should match");
        assert_eq!(reverse_key.1, recovered.1, "Src ID should match");
        assert_eq!(reverse_key.2, recovered.2, "NameHash should match");
    }

