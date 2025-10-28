// Database library for motlie

use uuid::Uuid;

mod writer;
pub use writer::*;

mod mutation;
pub use mutation::*;

mod graph;
pub use graph::*;

mod fulltext;
pub use fulltext::*;

/// A typesafe wrapper for UUID version 4
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Id(Uuid);

impl Id {
    /// Generate a new random UUID v4
    pub fn new() -> Self {
        Id(Uuid::new_v4())
    }

    /// Create from an existing Uuid
    pub fn from_uuid(uuid: Uuid) -> Self {
        Id(uuid)
    }

    /// Parse from a string, returning an error if invalid
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Id(Uuid::parse_str(s)?))
    }

    /// Get the string representation
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }

    /// Convert to String
    pub fn into_string(self) -> String {
        self.0.to_string()
    }

    /// Get the underlying Uuid
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Convert to the underlying Uuid
    pub fn into_uuid(self) -> Uuid {
        self.0
    }

    /// Check if this is a nil UUID (all zeros)
    pub fn is_nil(&self) -> bool {
        self.0.is_nil()
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<String> for Id {
    type Error = uuid::Error;

    fn try_from(uuid: String) -> Result<Self, Self::Error> {
        Id::from_str(&uuid)
    }
}

impl TryFrom<&str> for Id {
    type Error = uuid::Error;

    fn try_from(uuid: &str) -> Result<Self, Self::Error> {
        Id::from_str(uuid)
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Id> for String {
    fn from(uuid: Id) -> Self {
        uuid.into_string()
    }
}

#[derive(Debug, Clone)]
pub enum Mutation {
    AddVertex(AddVertexArgs),
    AddEdge(AddEdgeArgs),
    AddFragment(AddFragmentArgs),
    Invalidate(InvalidateArgs),
}

#[derive(Debug, Clone)]
pub struct AddVertexArgs {
    /// The UUID of the Vertex
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The name of the Vertex
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddEdgeArgs {
    /// The UUID of the source Vertex
    pub source_vertex_id: Id,

    /// The UUID of the target Vertex
    pub target_vertex_id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The name of the Edge
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddFragmentArgs {
    /// The UUID of the Vertex, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The body of the Fragment
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct InvalidateArgs {
    /// The UUID of the Vertex, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The reason for invalidation
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let graph_handle = spawn_graph_consumer(receiver1, config.clone());
        let fulltext_handle = spawn_fulltext_consumer(receiver2, config.clone());

        // Send mutations to both writers (simulating fanout)
        for i in 0..3 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("integration_test_vertex_{}", i),
            };

            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!("Integration test fragment {} with searchable content for both Graph storage and FullText indexing", i),
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
    fn test_uuid_new() {
        let uuid1 = Id::new();
        let uuid2 = Id::new();

        // UUIDs should be different
        assert_ne!(uuid1, uuid2);

        // Should not be nil
        assert!(!uuid1.is_nil());
        assert!(!uuid2.is_nil());
    }

    #[test]
    fn test_uuid_from_str_valid() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = Id::from_str(uuid_str).unwrap();

        assert_eq!(uuid.as_str(), uuid_str);
    }

    #[test]
    fn test_uuid_from_str_invalid() {
        let invalid_str = "not-a-uuid";
        assert!(Id::from_str(invalid_str).is_err());
    }

    #[test]
    fn test_uuid_display() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = Id::from_str(uuid_str).unwrap();

        assert_eq!(format!("{}", uuid), uuid_str);
    }

    #[test]
    fn test_uuid_try_from_string() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = Id::try_from(uuid_str.to_string()).unwrap();

        assert_eq!(uuid.as_str(), uuid_str);

        let invalid = Id::try_from("invalid".to_string());
        assert!(invalid.is_err());
    }

    #[test]
    fn test_uuid_try_from_str() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let uuid = Id::try_from(uuid_str).unwrap();

        assert_eq!(uuid.as_str(), uuid_str);

        let invalid = Id::try_from("invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_uuid_conversions() {
        let original_uuid = Uuid::new_v4();
        let uuid_v4 = Id::from_uuid(original_uuid);

        assert_eq!(uuid_v4.as_uuid(), &original_uuid);
        assert_eq!(uuid_v4.clone().into_uuid(), original_uuid);
        assert_eq!(uuid_v4.as_str(), original_uuid.to_string());
        assert_eq!(uuid_v4.into_string(), original_uuid.to_string());
    }

    #[test]
    fn test_uuid_default() {
        let uuid = Id::default();
        assert!(!uuid.is_nil());
    }

    #[test]
    fn test_uuid_nil() {
        let nil_uuid = Id::from_uuid(Uuid::nil());
        assert!(nil_uuid.is_nil());

        let normal_uuid = Id::new();
        assert!(!normal_uuid.is_nil());
    }

    #[test]
    fn test_uuid_string_conversion() {
        let uuid = Id::new();
        let uuid_string: String = uuid.clone().into();

        assert_eq!(uuid_string, uuid.as_str());
    }

    #[test]
    fn test_struct_usage() {
        // Test that our structs work with the new UuidV4 type
        let vertex = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };

        let edge = AddEdgeArgs {
            source_vertex_id: Id::new(),
            target_vertex_id: Id::new(),
            ts_millis: 1234567890,
            name: "test_edge".to_string(),
        };

        let fragment = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            body: "test fragment body".to_string(),
        };

        // Ensure they can be created and debugged
        println!("{:?}", vertex);
        println!("{:?}", edge);
        println!("{:?}", fragment);
    }
}
