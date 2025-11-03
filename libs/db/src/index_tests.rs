#[cfg(test)]
mod tests {
    use crate::index::{Edges, Fragments, Nodes, RowsToWrite};
    use crate::{AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id};

    #[test]
    fn test_nodes_plan_creates_single_row() {
        let id = Id::new();
        let args = AddVertexArgs {
            id,
            ts_millis: 1234567890,
            name: "TestVertex".to_string(),
        };

        let rows = Nodes::plan(&args);

        // Should create exactly one row
        assert_eq!(rows.0.len(), 1);

        // Verify key is the ID bytes
        assert_eq!(rows.0[0].0, id.into_bytes().to_vec());

        // Verify value contains expected markdown format
        let value = String::from_utf8(rows.0[0].1.clone()).unwrap();
        assert!(value.contains(&format!("id={}", id)));
        assert!(value.contains("TestVertex"));
        assert!(value.starts_with("[comment]:\\#"));
    }

    #[test]
    fn test_nodes_plan_preserves_node_name() {
        let test_cases = vec![
            "Simple Name",
            "Name with special chars: @#$%",
            "Name\nwith\nnewlines",
            "",
        ];

        for name in test_cases {
            let args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: name.to_string(),
            };

            let rows = Nodes::plan(&args);
            let value = String::from_utf8(rows.0[0].1.clone()).unwrap();

            assert!(
                value.contains(name),
                "Value should contain name '{}', got: {}",
                name,
                value
            );
        }
    }

    #[test]
    fn test_edges_plan_creates_two_rows() {
        let edge_id = Id::new();
        let source_id = Id::new();
        let target_id = Id::new();

        let args = AddEdgeArgs {
            id: edge_id,
            source_vertex_id: source_id,
            target_vertex_id: target_id,
            ts_millis: 1234567890,
            name: "TestEdge".to_string(),
        };

        let rows = Edges::plan(&args);

        // Should create exactly two rows
        assert_eq!(rows.0.len(), 2);
    }

    #[test]
    fn test_edges_plan_first_row_format() {
        let edge_id = Id::new();
        let source_id = Id::new();
        let target_id = Id::new();
        let edge_name = "TestEdge";

        let args = AddEdgeArgs {
            id: edge_id,
            source_vertex_id: source_id,
            target_vertex_id: target_id,
            ts_millis: 1234567890,
            name: edge_name.to_string(),
        };

        let rows = Edges::plan(&args);

        // First row: key = source_id + target_id + name, value = edge_id
        let row1 = &rows.0[0];

        // Verify key composition
        let mut expected_key = Vec::new();
        expected_key.extend_from_slice(&source_id.into_bytes());
        expected_key.extend_from_slice(&target_id.into_bytes());
        expected_key.extend_from_slice(edge_name.as_bytes());

        assert_eq!(row1.0, expected_key);

        // Verify value is the edge ID
        assert_eq!(row1.1, edge_id.into_bytes().to_vec());
    }

    #[test]
    fn test_edges_plan_second_row_format() {
        let edge_id = Id::new();
        let source_id = Id::new();
        let target_id = Id::new();
        let edge_name = "TestEdge";

        let args = AddEdgeArgs {
            id: edge_id,
            source_vertex_id: source_id,
            target_vertex_id: target_id,
            ts_millis: 1234567890,
            name: edge_name.to_string(),
        };

        let rows = Edges::plan(&args);

        // Second row: key = edge_id, value = markdown summary
        let row2 = &rows.0[1];

        // Verify key is edge ID
        assert_eq!(row2.0, edge_id.into_bytes().to_vec());

        // Verify value contains markdown summary
        let value = String::from_utf8(row2.1.clone()).unwrap();
        assert!(value.contains(&format!("id={}", edge_id)));
        assert!(value.contains(edge_name));
        assert!(value.starts_with("[comment]:\\#"));
    }

    #[test]
    fn test_edges_plan_different_edge_names() {
        let test_names = vec!["connects", "depends_on", "child_of", ""];

        for name in test_names {
            let args = AddEdgeArgs {
                id: Id::new(),
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: name.to_string(),
            };

            let rows = Edges::plan(&args);

            // Verify first row key contains the name
            assert!(rows.0[0].0.ends_with(name.as_bytes()));

            // Verify second row value contains the name
            let value = String::from_utf8(rows.0[1].1.clone()).unwrap();
            assert!(value.contains(name));
        }
    }

    #[test]
    fn test_fragments_plan_creates_single_row() {
        let fragment_id = Id::new();
        let timestamp: u64 = 1234567890;

        let args = AddFragmentArgs {
            id: fragment_id,
            ts_millis: timestamp,
            content: "Test fragment body".to_string(),
        };

        let rows = Fragments::plan(&args);

        // Should create exactly one row
        assert_eq!(rows.0.len(), 1);

        // Verify key is id + timestamp
        let mut expected_key = Vec::from(fragment_id.into_bytes());
        expected_key.extend_from_slice(&timestamp.to_be_bytes());

        assert_eq!(rows.0[0].0, expected_key);

        // Verify value contains the body
        let value = String::from_utf8(rows.0[0].1.clone()).unwrap();
        assert!(value.contains("Test fragment body"));
        assert!(value.contains(&format!("id={}", fragment_id)));
        assert!(value.starts_with("[comment]:\\#"));
    }

    #[test]
    fn test_fragments_plan_preserves_body_content() {
        let test_bodies = vec![
            "Simple text",
            "Text with\nnewlines\nand\ntabs\t",
            "Special chars: @#$%^&*()",
            "Unicode: 日本語 🎉",
            "",
        ];

        for body in test_bodies {
            let args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                content: body.to_string(),
            };

            let rows = Fragments::plan(&args);
            let value = String::from_utf8(rows.0[0].1.clone()).unwrap();

            assert!(
                value.contains(body),
                "Value should contain body '{}', got: {}",
                body,
                value
            );
        }
    }

    #[test]
    fn test_fragments_plan_timestamp_ordering() {
        let fragment_id = Id::new();
        let timestamps = vec![1000u64, 2000, 3000, 4000, 5000];

        let mut keys = Vec::new();

        for ts in timestamps {
            let args = AddFragmentArgs {
                id: fragment_id,
                ts_millis: ts,
                content: "Test".to_string(),
            };

            let rows = Fragments::plan(&args);
            keys.push(rows.0[0].0.clone());
        }

        // Verify keys are in ascending order (because of big-endian timestamp encoding)
        for i in 0..keys.len() - 1 {
            assert!(
                keys[i] < keys[i + 1],
                "Keys should be in ascending order by timestamp"
            );
        }
    }

    #[test]
    fn test_rows_to_write_structure() {
        // Test that RowsToWrite can hold multiple rows
        let rows = RowsToWrite(vec![
            (vec![1, 2, 3], vec![4, 5, 6]),
            (vec![7, 8, 9], vec![10, 11, 12]),
        ]);

        assert_eq!(rows.0.len(), 2);
        assert_eq!(rows.0[0].0, vec![1, 2, 3]);
        assert_eq!(rows.0[0].1, vec![4, 5, 6]);
        assert_eq!(rows.0[1].0, vec![7, 8, 9]);
        assert_eq!(rows.0[1].1, vec![10, 11, 12]);
    }

    #[test]
    fn test_nodes_plan_with_multiple_ids() {
        // Verify that different IDs produce different keys
        let id1 = Id::new();
        let id2 = Id::new();

        let args1 = AddVertexArgs {
            id: id1,
            ts_millis: 1234567890,
            name: "Vertex1".to_string(),
        };

        let args2 = AddVertexArgs {
            id: id2,
            ts_millis: 1234567890,
            name: "Vertex2".to_string(),
        };

        let rows1 = Nodes::plan(&args1);
        let rows2 = Nodes::plan(&args2);

        // Keys should be different
        assert_ne!(rows1.0[0].0, rows2.0[0].0);

        // But both should be valid ID bytes
        assert_eq!(rows1.0[0].0.len(), 16);
        assert_eq!(rows2.0[0].0.len(), 16);
    }

    #[test]
    fn test_edges_plan_symmetry() {
        // Test edge from A->B vs B->A
        let id_a = Id::new();
        let id_b = Id::new();
        let edge_id1 = Id::new();
        let edge_id2 = Id::new();

        let args1 = AddEdgeArgs {
            id: edge_id1,
            source_vertex_id: id_a,
            target_vertex_id: id_b,
            ts_millis: 1234567890,
            name: "connects".to_string(),
        };

        let args2 = AddEdgeArgs {
            id: edge_id2,
            source_vertex_id: id_b,
            target_vertex_id: id_a,
            ts_millis: 1234567890,
            name: "connects".to_string(),
        };

        let rows1 = Edges::plan(&args1);
        let rows2 = Edges::plan(&args2);

        // First row keys should be different (different source/target order)
        assert_ne!(rows1.0[0].0, rows2.0[0].0);

        // But both should have valid structure
        assert_eq!(rows1.0.len(), 2);
        assert_eq!(rows2.0.len(), 2);
    }

    #[test]
    fn test_fragments_plan_same_id_different_timestamps() {
        let fragment_id = Id::new();

        let args1 = AddFragmentArgs {
            id: fragment_id,
            ts_millis: 1000,
            content: "First fragment".to_string(),
        };

        let args2 = AddFragmentArgs {
            id: fragment_id,
            ts_millis: 2000,
            content: "Second fragment".to_string(),
        };

        let rows1 = Fragments::plan(&args1);
        let rows2 = Fragments::plan(&args2);

        // Keys should be different due to different timestamps
        assert_ne!(rows1.0[0].0, rows2.0[0].0);

        // But both should start with the same ID
        assert_eq!(
            &rows1.0[0].0[..16],
            &rows2.0[0].0[..16],
            "First 16 bytes (ID) should be the same"
        );

        // Timestamps should be in the key
        assert_eq!(rows1.0[0].0.len(), 16 + 8); // ID + u64 timestamp
        assert_eq!(rows2.0[0].0.len(), 16 + 8);
    }
}
