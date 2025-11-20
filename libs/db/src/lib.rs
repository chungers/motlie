use ferroid::base32::Base32UlidExt;
use ferroid::id::ULID;

mod writer;
use serde::{Deserialize, Serialize};
pub use writer::*;

mod mutation;
// Re-export mutation types
pub use mutation::{
    spawn_consumer as spawn_mutation_consumer, AddEdge, AddEdgeFragment, AddNode, AddNodeFragment,
    Consumer as MutationConsumer, Mutation, MutationBatch, MutationPlanner,
    Processor as MutationProcessor, Runnable as MutationRunnable, UpdateEdgeValidSinceUntil,
    UpdateEdgeWeight, UpdateNodeValidSinceUntil,
};
// Note: mutations![] macro is automatically available via #[macro_export] in mutation.rs

mod reader;
pub use reader::*;

mod query;
// Re-export query types and consumer functions
pub use query::{
    EdgeSummaryBySrcDstName, EdgesByName, IncomingEdges, NodeById, NodeFragmentsByIdTimeRange,
    NodesByName, OutgoingEdges, Query, Runnable as QueryRunnable,
    Processor as QueryProcessor, Consumer as QueryConsumer, spawn_consumer as spawn_query_consumer,
};
pub use schema::{DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, SrcId};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TimestampMilli(pub u64);

impl TimestampMilli {
    /// Create a new timestamp from the current time
    pub fn now() -> Self {
        TimestampMilli(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        )
    }
}

/// A wrapper around a data URL string that provides encoding/decoding capabilities.
///
/// Format: `data:<mime_type>;charset=utf-8;base64,<base64-encoded-content>`
///
/// This is OpenAI-compliant and can be used for text, markdown, JSON, images, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataUrl(String);

/// Custom error type for DataUrl operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataUrlError(String);

impl std::fmt::Display for DataUrlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataUrl error: {}", self.0)
    }
}

impl std::error::Error for DataUrlError {}

/// Macro to generate file-loading methods for different image formats
macro_rules! impl_from_file {
    ($method_name:ident, $mime_type:expr, $doc:expr) => {
        #[doc = $doc]
        ///
        /// # Arguments
        /// * `path` - Path to the file to load
        ///
        /// # Returns
        /// * `Result<DataUrl, std::io::Error>` - The DataUrl on success, or an IO error
        ///
        /// # Example
        /// ```ignore
        #[doc = concat!("let data_url = DataUrl::", stringify!($method_name), r#"("image.jpg")?;"#)]
        /// ```
        pub fn $method_name(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
            let bytes = std::fs::read(path)?;
            Ok(Self::new(&bytes, $mime_type))
        }
    };
}

impl DataUrl {
    /// Create a new DataUrl from raw content with the specified MIME type
    fn new(content: impl AsRef<[u8]>, mime_type: &str) -> Self {
        use base64::{engine::general_purpose, Engine as _};
        let encoded = general_purpose::STANDARD.encode(content.as_ref());
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        DataUrl(data_url)
    }

    /// Create a DataUrl for markdown content
    pub fn from_markdown(content: impl AsRef<str>) -> Self {
        Self::new(content.as_ref().as_bytes(), "text/markdown;charset=utf-8")
    }

    /// Create a DataUrl for plain text content
    pub fn from_text(content: impl AsRef<str>) -> Self {
        Self::new(content.as_ref().as_bytes(), "text/plain;charset=utf-8")
    }

    /// Create a DataUrl for JSON content
    pub fn from_json(content: impl AsRef<str>) -> Self {
        Self::new(
            content.as_ref().as_bytes(),
            "application/json;charset=utf-8",
        )
    }

    /// Create a DataUrl for HTML content
    pub fn from_html(content: impl AsRef<str>) -> Self {
        Self::new(content.as_ref().as_bytes(), "text/html;charset=utf-8")
    }

    /// Create a DataUrl for JPEG image data
    ///
    /// # Example
    /// ```ignore
    /// let image_bytes = std::fs::read("image.jpg")?;
    /// let data_url = DataUrl::from_jpeg(&image_bytes);
    /// ```
    pub fn from_jpeg(image_data: impl AsRef<[u8]>) -> Self {
        Self::new(image_data.as_ref(), "image/jpeg")
    }

    /// Create a DataUrl for PNG image data
    pub fn from_png(image_data: impl AsRef<[u8]>) -> Self {
        Self::new(image_data.as_ref(), "image/png")
    }

    // Generate file-loading methods using the macro
    impl_from_file!(
        from_png_file,
        "image/png",
        "Create a DataUrl by loading a PNG file from disk"
    );

    impl_from_file!(
        from_jpeg_file,
        "image/jpeg",
        "Create a DataUrl by loading a JPEG file from disk"
    );

    impl_from_file!(
        from_text_file,
        "text/plain;charset=utf-8",
        "Create a DataUrl by loading a plain text file from disk"
    );

    impl_from_file!(
        from_markdown_file,
        "text/markdown;charset=utf-8",
        "Create a DataUrl by loading a Markdown file from disk"
    );

    impl_from_file!(
        from_html_file,
        "text/html;charset=utf-8",
        "Create a DataUrl by loading an HTML file from disk"
    );

    impl_from_file!(
        from_json_file,
        "application/json;charset=utf-8",
        "Create a DataUrl by loading a JSON file from disk"
    );

    /// Decode and extract the content as bytes (useful for binary data like images)
    pub fn decode_bytes(&self) -> Result<Vec<u8>, DataUrlError> {
        let parsed = data_url::DataUrl::process(&self.0)
            .map_err(|e| DataUrlError(format!("Parse error: {}", e)))?;

        let (body, _fragment) = parsed
            .decode_to_vec()
            .map_err(|e| DataUrlError(format!("Decode error: {}", e)))?;

        Ok(body)
    }

    /// Decode and extract the content as a UTF-8 string (for text-based content)
    pub fn decode_string(&self) -> Result<String, DataUrlError> {
        let bytes = self.decode_bytes()?;
        String::from_utf8(bytes).map_err(|e| DataUrlError(format!("UTF-8 error: {}", e)))
    }

    /// Extract the MIME type from the data URL
    pub fn mime_type(&self) -> Result<String, DataUrlError> {
        let parsed = data_url::DataUrl::process(&self.0)
            .map_err(|e| DataUrlError(format!("Parse error: {}", e)))?;
        Ok(parsed.mime_type().to_string())
    }
}

impl std::fmt::Display for DataUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<DataUrl> for String {
    fn from(data_url: DataUrl) -> String {
        data_url.0
    }
}

impl AsRef<str> for DataUrl {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

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

    /// Returns the timestamp of the ULID
    pub fn timestamp(&self) -> TimestampMilli {
        TimestampMilli(ULID::from_raw(u128::from_be_bytes(self.0)).timestamp() as u64)
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
    use crate::mutation::Runnable as MutRunnable;
    use std::path::Path;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_graph_and_fulltext_consumers_integration() {
        // Test that both Graph and FullText consumers can process mutations from the same writer
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        // Create temporary directory for test database (auto-cleanup on drop)
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Create writer and two separate receivers for each consumer type
        let (writer1, receiver1) = create_mutation_writer(config.clone());
        let (writer2, receiver2) = create_mutation_writer(config.clone());

        // Spawn both consumer types
        let graph_handle = spawn_graph_consumer(receiver1, config.clone(), temp_dir.path());
        let fulltext_handle = spawn_fulltext_consumer(receiver2, config.clone());

        // Send mutations to both writers (simulating fanout)
        for i in 0..3 {
            let node_id = Id::new();
            let node_args = AddNode {
                id: node_id.clone(),
                ts_millis: TimestampMilli::now(),
                name: format!("integration_test_node_{}", i),
                temporal_range: None,
            };

            let fragment_args = AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text(&format!("Integration test fragment {} with searchable content for both Graph storage and FullText indexing", i)),
                temporal_range: None,
            };

            // Send to both consumers
            node_args.clone().run(&writer1).await.unwrap();
            fragment_args.clone().run(&writer1).await.unwrap();

            node_args.run(&writer2).await.unwrap();
            fragment_args.run(&writer2).await.unwrap();
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
        let node = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
        };

        let edge = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
            summary: EdgeSummary::from_text("edge summary"),
            weight: Some(1.0),
            temporal_range: None,
        };

        let fragment = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("test fragment body"),
            temporal_range: None,
        };

        // Ensure they can be created and debugged
        println!("{:?}", node);
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

    // DataUrl tests
    #[test]
    fn test_data_url_from_markdown() {
        let content = "# Hello World\nThis is **markdown** with _formatting_.";
        let data_url = DataUrl::from_markdown(content);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:text/markdown"));
        assert_eq!(data_url.decode_string().unwrap(), content);

        let mime = data_url.mime_type().unwrap();
        assert!(mime.contains("markdown"));
    }

    #[test]
    fn test_data_url_from_text() {
        let content = "Plain text content without formatting";
        let data_url = DataUrl::from_text(content);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:text/plain"));
        assert_eq!(data_url.decode_string().unwrap(), content);

        let mime = data_url.mime_type().unwrap();
        assert!(mime.contains("text/plain"));
    }

    #[test]
    fn test_data_url_from_json() {
        let content = r#"{"key": "value", "nested": {"foo": "bar"}}"#;
        let data_url = DataUrl::from_json(content);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:application/json"));
        assert_eq!(data_url.decode_string().unwrap(), content);

        let mime = data_url.mime_type().unwrap();
        assert!(mime.contains("application/json"));
    }

    #[test]
    fn test_data_url_from_html() {
        let content = "<html><body><h1>Hello</h1><p>World</p></body></html>";
        let data_url = DataUrl::from_html(content);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:text/html"));
        assert_eq!(data_url.decode_string().unwrap(), content);

        let mime = data_url.mime_type().unwrap();
        assert!(mime.contains("text/html"));
    }

    #[test]
    fn test_data_url_from_jpeg() {
        // Simulate JPEG binary data (actual JPEG header + some data)
        let jpeg_data = vec![
            0xFF, 0xD8, 0xFF, 0xE0, // JPEG header
            0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, // JFIF marker
            0x01, 0x02, 0x03, 0x04, 0x05, // dummy data
        ];

        let data_url = DataUrl::from_jpeg(&jpeg_data);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:image/jpeg"));

        let decoded = data_url.decode_bytes().unwrap();
        assert_eq!(decoded, jpeg_data);

        let mime = data_url.mime_type().unwrap();
        assert_eq!(mime, "image/jpeg");
    }

    #[test]
    fn test_data_url_from_png() {
        // PNG signature
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
            0x00, 0x01, 0x02, 0x03, // dummy data
        ];

        let data_url = DataUrl::from_png(&png_data);

        let data_str: &str = data_url.as_ref();
        assert!(data_str.starts_with("data:image/png"));

        let decoded = data_url.decode_bytes().unwrap();
        assert_eq!(decoded, png_data);

        let mime = data_url.mime_type().unwrap();
        assert_eq!(mime, "image/png");
    }

    #[test]
    fn test_data_url_binary_roundtrip() {
        // Test with various binary patterns
        let binary_patterns = vec![
            vec![0u8; 100],                    // All zeros
            vec![255u8; 100],                  // All ones
            (0u8..=255u8).collect::<Vec<_>>(), // Full byte range
        ];

        for binary in binary_patterns {
            let data_url = DataUrl::from_jpeg(&binary);
            let decoded = data_url.decode_bytes().unwrap();
            assert_eq!(decoded, binary, "Binary roundtrip should preserve data");
        }
    }

    #[test]
    fn test_data_url_roundtrip() {
        let original = "Test content with special chars: émojis 🎉 and symbols @#$%";
        let data_url = DataUrl::from_markdown(original);
        let decoded = data_url.decode_string().unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_data_url_mime_type_extraction() {
        let markdown_url = DataUrl::from_markdown("# Content");
        assert!(markdown_url.mime_type().unwrap().contains("markdown"));

        let text_url = DataUrl::from_text("content");
        assert!(text_url.mime_type().unwrap().contains("plain"));

        let json_url = DataUrl::from_json(r#"{"key": "value"}"#);
        assert!(json_url.mime_type().unwrap().contains("json"));

        let html_url = DataUrl::from_html("<p>test</p>");
        assert!(html_url.mime_type().unwrap().contains("html"));

        let jpeg_url = DataUrl::from_jpeg(&[0xFF, 0xD8]);
        assert_eq!(jpeg_url.mime_type().unwrap(), "image/jpeg");
    }

    #[test]
    fn test_data_url_display() {
        let data_url = DataUrl::from_text("Hello");
        let display_str = format!("{}", data_url);
        assert!(display_str.starts_with("data:"));
        assert!(display_str.contains("base64"));
    }

    #[test]
    fn test_data_url_as_ref_str() {
        let data_url = DataUrl::from_text("Hello World");
        let s: &str = data_url.as_ref();
        assert!(s.starts_with("data:text/plain"));
        assert!(s.contains("base64"));
    }

    #[test]
    fn test_data_url_into_string() {
        let data_url = DataUrl::from_text("Hello");
        let s: String = data_url.clone().into();
        assert!(s.starts_with("data:"));
    }

    #[test]
    fn test_data_url_serde_roundtrip() {
        let original = DataUrl::from_markdown("# Test\nContent");
        let serialized = rmp_serde::to_vec(&original).expect("Failed to serialize");
        let deserialized: DataUrl =
            rmp_serde::from_slice(&serialized).expect("Failed to deserialize");
        assert_eq!(original, deserialized);
        assert_eq!(
            original.decode_string().unwrap(),
            deserialized.decode_string().unwrap()
        );
    }

    #[test]
    fn test_data_url_empty_string() {
        let data_url = DataUrl::from_text("");
        assert_eq!(data_url.decode_string().unwrap(), "");
    }

    #[test]
    fn test_data_url_multiline_content() {
        let multiline = "Line 1\nLine 2\nLine 3\n\nLine 5";
        let data_url = DataUrl::from_markdown(multiline);
        assert_eq!(data_url.decode_string().unwrap(), multiline);
    }

    #[test]
    fn test_data_url_unicode_content() {
        let unicode = "Hello 世界 🌍 مرحبا שלום";
        let data_url = DataUrl::from_text(unicode);
        assert_eq!(data_url.decode_string().unwrap(), unicode);
    }

    #[test]
    fn test_data_url_from_png_file() {
        use std::io::Write;

        // Create a temporary PNG file with a minimal valid PNG header
        let temp_dir = tempfile::TempDir::new().unwrap();
        let png_path = temp_dir.path().join("test.png");

        // Minimal valid PNG file (1x1 pixel, white)
        let png_data: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
            0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F, 0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC,
            0xCC, 0x59, 0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];

        std::fs::File::create(&png_path)
            .unwrap()
            .write_all(png_data)
            .unwrap();

        // Test loading the file
        let data_url = DataUrl::from_png_file(&png_path).unwrap();

        // Verify it's a proper data URL
        assert!(data_url.0.starts_with("data:image/png;base64,"));

        // Verify we can decode it back to the same bytes
        let decoded = data_url.decode_bytes().unwrap();
        assert_eq!(decoded, png_data);

        // Verify MIME type
        assert_eq!(data_url.mime_type().unwrap(), "image/png");
    }

    #[test]
    fn test_data_url_from_jpeg_file() {
        use std::io::Write;

        // Create a temporary JPEG file with a minimal valid JPEG header
        let temp_dir = tempfile::TempDir::new().unwrap();
        let jpeg_path = temp_dir.path().join("test.jpg");

        // Minimal valid JPEG file (placeholder - real JPEG would be more complex)
        let jpeg_data: &[u8] = &[
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, // SOI + APP0 marker
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF,
            0xD9, // EOI marker
        ];

        std::fs::File::create(&jpeg_path)
            .unwrap()
            .write_all(jpeg_data)
            .unwrap();

        // Test loading the file
        let data_url = DataUrl::from_jpeg_file(&jpeg_path).unwrap();

        // Verify it's a proper data URL
        assert!(data_url.0.starts_with("data:image/jpeg;base64,"));

        // Verify we can decode it back to the same bytes
        let decoded = data_url.decode_bytes().unwrap();
        assert_eq!(decoded, jpeg_data);

        // Verify MIME type
        assert_eq!(data_url.mime_type().unwrap(), "image/jpeg");
    }

    #[test]
    fn test_data_url_from_text_file() {
        use std::io::Write;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let text_path = temp_dir.path().join("test.txt");

        let text_content = "Hello, World!\nThis is a test file.";
        std::fs::File::create(&text_path)
            .unwrap()
            .write_all(text_content.as_bytes())
            .unwrap();

        let data_url = DataUrl::from_text_file(&text_path).unwrap();

        assert!(data_url
            .0
            .starts_with("data:text/plain;charset=utf-8;base64,"));
        assert_eq!(data_url.decode_string().unwrap(), text_content);
        assert_eq!(data_url.mime_type().unwrap(), "text/plain;charset=utf-8");
    }

    #[test]
    fn test_data_url_from_markdown_file() {
        use std::io::Write;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let md_path = temp_dir.path().join("test.md");

        let md_content = "# Hello\n\nThis is **markdown**.\n\n- Item 1\n- Item 2";
        std::fs::File::create(&md_path)
            .unwrap()
            .write_all(md_content.as_bytes())
            .unwrap();

        let data_url = DataUrl::from_markdown_file(&md_path).unwrap();

        assert!(data_url
            .0
            .starts_with("data:text/markdown;charset=utf-8;base64,"));
        assert_eq!(data_url.decode_string().unwrap(), md_content);
        assert_eq!(data_url.mime_type().unwrap(), "text/markdown;charset=utf-8");
    }

    #[test]
    fn test_data_url_from_html_file() {
        use std::io::Write;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let html_path = temp_dir.path().join("test.html");

        let html_content = r#"<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><h1>Hello</h1></body>
</html>"#;
        std::fs::File::create(&html_path)
            .unwrap()
            .write_all(html_content.as_bytes())
            .unwrap();

        let data_url = DataUrl::from_html_file(&html_path).unwrap();

        assert!(data_url
            .0
            .starts_with("data:text/html;charset=utf-8;base64,"));
        assert_eq!(data_url.decode_string().unwrap(), html_content);
        assert_eq!(data_url.mime_type().unwrap(), "text/html;charset=utf-8");
    }

    #[test]
    fn test_data_url_from_json_file() {
        use std::io::Write;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let json_path = temp_dir.path().join("test.json");

        let json_content = r#"{"name": "test", "value": 42, "active": true}"#;
        std::fs::File::create(&json_path)
            .unwrap()
            .write_all(json_content.as_bytes())
            .unwrap();

        let data_url = DataUrl::from_json_file(&json_path).unwrap();

        assert!(data_url
            .0
            .starts_with("data:application/json;charset=utf-8;base64,"));
        assert_eq!(data_url.decode_string().unwrap(), json_content);
        assert_eq!(
            data_url.mime_type().unwrap(),
            "application/json;charset=utf-8"
        );
    }

    #[test]
    fn test_data_url_from_file_nonexistent() {
        // Test that loading a nonexistent file returns an error
        let result = DataUrl::from_png_file("/nonexistent/path/to/file.png");
        assert!(result.is_err());

        let result = DataUrl::from_jpeg_file("/nonexistent/path/to/file.jpg");
        assert!(result.is_err());

        let result = DataUrl::from_text_file("/nonexistent/path/to/file.txt");
        assert!(result.is_err());

        let result = DataUrl::from_markdown_file("/nonexistent/path/to/file.md");
        assert!(result.is_err());

        let result = DataUrl::from_html_file("/nonexistent/path/to/file.html");
        assert!(result.is_err());

        let result = DataUrl::from_json_file("/nonexistent/path/to/file.json");
        assert!(result.is_err());
    }
}
