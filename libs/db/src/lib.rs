use ferroid::base32::Base32UlidExt;
use ferroid::id::ULID;

mod writer;
use serde::{Deserialize, Serialize};
pub use writer::*;
mod mutation;
pub use mutation::*;
mod graph;
pub use graph::*;
mod fulltext;
pub use fulltext::*;
mod schema;

#[cfg(test)]
mod fulltext_tests;
#[cfg(test)]
mod graph_tests;

/// Custom error type for Id parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdError(String);

impl std::fmt::Display for IdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid ID: {}", self.0)
    }
}

impl std::error::Error for IdError {}

/// A typesafe wrapper for ULID with 128-bit internal representation as 16 bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Id([u8; 16]);

impl Id {
    /// Generate a new ULID with 128-bit internal representation
    pub fn new() -> Self {
        let ulid = ULID::from_timestamp(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        );
        Id(ulid.to_raw().to_be_bytes())
    }

    /// Create from a byte slice
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Id(bytes)
    }

    /// Parse from a string, returning an error if invalid
    pub fn from_str(s: &str) -> Result<Self, IdError> {
        // Try to decode as ULID string first
        if let Ok(ulid) = ULID::decode(s) {
            return Ok(Id(ulid.to_raw().to_be_bytes()));
        }

        Err(IdError(format!("Could not parse '{}' as ULID", s)))
    }

    /// Get the string representation as ULID
    pub fn as_str(&self) -> String {
        let ulid = ULID::from_raw(u128::from_be_bytes(self.0));
        format!("{}", ulid.encode())
    }

    /// Convert to String
    pub fn into_string(self) -> String {
        self.as_str()
    }

    /// Get the underlying byte array
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Convert to the underlying byte array
    pub fn into_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Check if this is a nil ID (all zeros)
    pub fn is_nil(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Create a nil ID (all zeros)
    pub fn nil() -> Self {
        Id([0u8; 16])
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl TryFrom<String> for Id {
    type Error = IdError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Id::from_str(&s)
    }
}

impl TryFrom<&str> for Id {
    type Error = IdError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Id::from_str(s)
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Id> for String {
    fn from(id: Id) -> Self {
        id.into_string()
    }
}

impl From<[u8; 16]> for Id {
    fn from(bytes: [u8; 16]) -> Self {
        Id(bytes)
    }
}

impl From<Id> for [u8; 16] {
    fn from(id: Id) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_graph_and_fulltext_consumers_integration() {
        // Test that both Graph and FullText consumers can process mutations from the same writer
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        // Create writer and two separate receivers for each consumer type
        let (writer1, receiver1) = create_mutation_writer(config.clone());
        let (writer2, receiver2) = create_mutation_writer(config.clone());

        // Spawn both consumer types
        let graph_handle =
            spawn_graph_consumer(receiver1, config.clone(), Path::new("/tmp/test_graph_db"));
        let fulltext_handle = spawn_fulltext_consumer(receiver2, config.clone());

        // Send mutations to both writers (simulating fanout)
        for i in 0..3 {
            let vertex_args = AddNodeArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("integration_test_vertex_{}", i),
            };

            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                content: format!("Integration test fragment {} with searchable content for both Graph storage and FullText indexing", i),
            };

            // Send to both consumers
            writer1.add_vertex(vertex_args.clone()).await.unwrap();
            writer1.add_fragment(fragment_args.clone()).await.unwrap();

            writer2.add_vertex(vertex_args).await.unwrap();
            writer2.add_fragment(fragment_args).await.unwrap();
        }

        // Give consumers time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown both consumers
        drop(writer1);
        drop(writer2);

        // Wait for both consumers to complete
        graph_handle.await.unwrap().unwrap();
        fulltext_handle.await.unwrap().unwrap();
    }

    #[test]
    fn test_id_new() {
        let id1 = Id::new();
        let id2 = Id::new();
        // Should not be nil
        assert!(!id1.is_nil());
        assert!(!id2.is_nil());
    }

    #[test]
    fn test_id_from_bytes() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = Id::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        assert_eq!(id.into_bytes(), bytes);
    }

    #[test]
    fn test_id_from_str_ulid() {
        let ulid = ULID::from_timestamp(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        );
        let ulid_str = format!("{}", ulid.encode());
        let id = Id::from_str(&ulid_str).unwrap();
        let expected_bytes = ulid.to_raw().to_be_bytes();
        assert_eq!(id.as_bytes(), &expected_bytes);
    }

    #[test]
    fn test_id_from_str_invalid() {
        let invalid_str = "not-a-valid-id";
        assert!(Id::from_str(invalid_str).is_err());
    }

    #[test]
    fn test_id_display() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = Id::from_bytes(bytes);
        let display_str = format!("{}", id);
        // Should be able to parse it back
        let parsed_id = Id::from_str(&display_str).unwrap();
        assert_eq!(id, parsed_id);
    }

    #[test]
    fn test_id_try_from_string() {
        let ulid = ULID::from_timestamp(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        );
        let ulid_str = format!("{}", ulid.encode());

        let id = Id::try_from(ulid_str.clone()).unwrap();
        assert_eq!(id.as_bytes(), &ulid.to_raw().to_be_bytes());

        let invalid = Id::try_from("invalid".to_string());
        assert!(invalid.is_err());
    }

    #[test]
    fn test_id_try_from_str() {
        let ulid = ULID::from_timestamp(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        );
        let ulid_str = format!("{}", ulid.encode());

        let id = Id::try_from(ulid_str.as_str()).unwrap();
        assert_eq!(id.as_bytes(), &ulid.to_raw().to_be_bytes());

        let invalid = Id::try_from("invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_id_conversions() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = Id::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        assert_eq!(id.into_bytes(), bytes);

        // Test From trait
        let id2: Id = bytes.into();
        assert_eq!(id2.as_bytes(), &bytes);

        let bytes2: [u8; 16] = id2.into();
        assert_eq!(bytes2, bytes);
    }

    #[test]
    fn test_id_default() {
        let id = Id::default();
        assert!(!id.is_nil());
    }

    #[test]
    fn test_id_nil() {
        let nil_id = Id::nil();
        assert!(nil_id.is_nil());
        assert_eq!(nil_id.as_bytes(), &[0u8; 16]);

        let normal_id = Id::new();
        assert!(!normal_id.is_nil());
    }

    #[test]
    fn test_id_string_conversion() {
        let id = Id::new();
        let id_string: String = id.into();
        assert!(!id_string.is_empty());
    }

    #[test]
    fn test_id_ordering() {
        let bytes1 = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let bytes2 = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2];
        let id1 = Id::from_bytes(bytes1);
        let id2 = Id::from_bytes(bytes2);
        assert!(id1 < id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_id_lexicographic_sorting() {
        // Generate multiple IDs with small delays to ensure different timestamps
        let mut ids = Vec::new();
        for _ in 0..10 {
            ids.push(Id::new());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // Store the original order
        let original_ids = ids.clone();

        // Sort the IDs
        ids.sort();

        // The sorted order should be the same as the generated order
        // because ULIDs are lexicographically sortable by timestamp
        assert_eq!(
            ids, original_ids,
            "IDs should be lexicographically sorted in generation order"
        );

        // Verify each ID is less than or equal to the next
        for i in 0..ids.len() - 1 {
            assert!(
                ids[i] <= ids[i + 1],
                "ID at index {} should be <= ID at index {}",
                i,
                i + 1
            );
        }
    }

    #[test]
    fn test_id_sequential_generation() {
        // Generate IDs sequentially and verify they are in order
        let id1 = Id::new();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id2 = Id::new();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id3 = Id::new();

        // Each subsequent ID should be greater than the previous
        assert!(id1 < id2, "id1 should be less than id2");
        assert!(id2 < id3, "id2 should be less than id3");
        assert!(id1 < id3, "id1 should be less than id3");

        // Verify the byte-level ordering
        assert!(id1.as_bytes() < id2.as_bytes());
        assert!(id2.as_bytes() < id3.as_bytes());
    }

    #[test]
    fn test_id_byte_slice_lexicographic_order() {
        // Test that byte slice comparison matches Id comparison
        let mut ids = Vec::new();
        for _ in 0..5 {
            ids.push(Id::new());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // Compare using Id's Ord implementation
        let mut sorted_ids = ids.clone();
        sorted_ids.sort();

        // Compare using byte slice comparison
        let mut byte_sorted: Vec<_> = ids.iter().map(|id| id.as_bytes().clone()).collect();
        byte_sorted.sort();

        // The orderings should match
        for (i, id) in sorted_ids.iter().enumerate() {
            assert_eq!(
                id.as_bytes(),
                &byte_sorted[i],
                "Byte ordering should match Id ordering at index {}",
                i
            );
        }
    }

    #[test]
    fn test_struct_usage() {
        // Test that our structs work with the new Id type
        let vertex = AddNodeArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };

        let edge = AddEdgeArgs {
            id: Id::new(),
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: 1234567890,
            name: "test_edge".to_string(),
        };

        let fragment = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            content: "test fragment body".to_string(),
        };

        // Ensure they can be created and debugged
        println!("{:?}", vertex);
        println!("{:?}", edge);
        println!("{:?}", fragment);
    }

    #[test]
    fn test_id_serde_matches_into_bytes() {
        // Create an Id with known bytes
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id = Id::from_bytes(bytes);

        // Serialize using MessagePack
        let serialized = rmp_serde::to_vec(&id).expect("Failed to serialize Id");

        // Get the bytes using into_bytes()
        let id_bytes = id.into_bytes();

        // The serialized version should match the raw bytes
        // MessagePack encodes a 16-byte array as: 0xC4 (fixext type) + 0x10 (length 16) + 16 bytes
        // or as 0xDC (array16) + length + elements if it's an array
        // Since Id is defined as Id([u8; 16]), serde will serialize it as a byte array
        // MessagePack byte arrays are encoded as binary format: 0xC4 + len + data (for small arrays)

        // Let's verify by deserializing back
        let deserialized: Id =
            rmp_serde::from_slice(&serialized).expect("Failed to deserialize Id");
        assert_eq!(
            deserialized.into_bytes(),
            id_bytes,
            "Deserialized Id should match original bytes"
        );

        // Now verify that the serialized format contains our bytes
        // For a [u8; 16], MessagePack will encode as binary data
        // The format should be: [type byte(s), length info, actual 16 bytes]
        assert!(
            serialized.len() >= 16,
            "Serialized data should contain at least the 16 bytes"
        );

        // Extract the actual data bytes from the MessagePack format
        // and verify they match id_bytes
        assert!(
            serialized.ends_with(&id_bytes) || serialized[serialized.len() - 16..] == id_bytes,
            "Serialized data should end with the same bytes as into_bytes()"
        );
    }

    #[test]
    fn test_id_serde_roundtrip() {
        // Test serialization and deserialization roundtrip
        let original_id = Id::new();
        let original_bytes = original_id.into_bytes();

        // Serialize
        let serialized = rmp_serde::to_vec(&original_id).expect("Failed to serialize");

        // Deserialize
        let deserialized: Id = rmp_serde::from_slice(&serialized).expect("Failed to deserialize");
        let deserialized_bytes = deserialized.into_bytes();

        // Both methods should produce the same bytes
        assert_eq!(
            original_bytes, deserialized_bytes,
            "Roundtrip should preserve bytes"
        );
    }
}
