//! Mutation writer module providing mutation infrastructure.
//!
//! This module follows the same pattern as fulltext::writer:
//! - Writer - handle for sending mutations
//! - WriterConfig - configuration
//! - Consumer - processes mutations from channel
//! - Spawn functions for creating consumers
//!
//! Also contains the mutation executor traits (MutationExecutor, Processor)
//! which define how mutations execute against the storage layer.
//!
//! # Transaction Support
//!
//! The `Writer` type also supports creating transactions for read-your-writes
//! semantics via the `transaction()` method. See the [`transaction`](super::transaction)
//! module for details.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use super::mutation::{ExecOptions, FlushMarker, Mutation, MutationResult};
use super::name_hash::NameCache;
use super::processor::Processor as GraphProcessor;
use super::transaction::Transaction;
use super::Storage;
use crate::request::{new_request_id, ReplyEnvelope, RequestEnvelope};

// ============================================================================
// MutationExecutor Trait
// ============================================================================

/// Trait for mutations to execute themselves directly against storage.
///
/// This trait defines HOW to write the mutation to the database.
/// Each mutation type knows how to execute its own database write operations.
///
/// Following the same pattern as QueryExecutor for queries.
/// Note: This is synchronous because RocksDB operations are blocking.
pub trait MutationExecutor: Send + Sync {
    /// Execute this mutation directly against a RocksDB transaction.
    /// Each mutation type knows how to write itself to storage.
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()>;

    /// Execute this mutation with runtime options, returning a reply payload.
    ///
    /// Default implementation delegates to `execute()` and returns an empty reply.
    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        self.execute(txn, txn_db)?;
        Ok(MutationResult::Flush)
    }

    /// Execute this mutation with access to the name cache.
    ///
    /// The cache is used to:
    /// 1. Skip redundant Names CF writes for already-interned names
    /// 2. Intern new names for future lookups
    ///
    /// Default implementation delegates to `execute()` (ignoring the cache).
    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &NameCache,
    ) -> Result<()> {
        self.execute(txn, txn_db)
    }

    /// Execute with cache and runtime options, returning a reply payload.
    ///
    /// Default implementation ignores cache and delegates to execute_with_options().
    fn execute_with_cache_and_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &NameCache,
        options: ExecOptions,
    ) -> Result<MutationResult> {
        self.execute_with_options(txn, txn_db, options)
    }
}

// ============================================================================
// Processor Trait
// ============================================================================

/// Trait for processing batches of mutations atomically.
///
/// Consumers delegate to a Processor to handle the actual database operations.
/// This separation allows:
/// - Different storage backends (RocksDB, Tantivy, etc.)
/// - Multiple consumers to process the same mutations
/// - Testing with mock processors
///
/// # Example Implementation
///
/// ```rust,ignore
/// #[async_trait::async_trait]
/// impl Processor for graph::Processor {
///     async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
///         // Execute mutations in a single RocksDB transaction
///         let txn_db = self.storage().transaction_db()?;
///         let txn = txn_db.transaction();
///
///         for mutation in mutations {
///             mutation.execute_with_cache(&txn, txn_db, self.name_cache())?;
///         }
///
///         txn.commit()?;  // Single commit for entire batch
///         Ok(())
///     }
/// }
/// ```
// (claude, 2026-02-07, FIXED: Updated example to use graph::Processor instead of Graph per codex eval)
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process a batch of mutations atomically.
    ///
    /// # Arguments
    /// * `mutations` - Slice of mutations to process. Can be a single mutation or many.
    ///
    /// # Returns
    /// * `Ok(())` if all mutations were processed successfully
    /// * `Err(_)` if processing failed (implementations should rollback on error)
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;

    /// Process a batch of mutations with execution options, returning per-mutation replies.
    async fn process_mutations_with_options(
        &self,
        mutations: &[Mutation],
        _options: ExecOptions,
    ) -> Result<Vec<MutationResult>> {
        self.process_mutations(mutations).await?;
        Ok(vec![MutationResult::Flush; mutations.len()])
    }
}

// ============================================================================
// Writer
// ============================================================================

/// Configuration for the mutation writer
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Size of the MPSC channel buffer
    pub channel_buffer_size: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
        }
    }
}

// ============================================================================
// MutationRequest
// ============================================================================

/// Envelope for mutation batches with execution options and optional reply.
pub type MutationRequest = RequestEnvelope<Vec<Mutation>>;

/// Handle for sending mutations to the writer with batching support.
///
/// The `Writer` sends mutations through an MPSC channel as `Vec<Mutation>` to enable
/// efficient transaction batching in downstream processors.
///
/// # Usage
///
/// Use the new mutation API for sending mutations:
///
/// ## Single Mutations
///
/// ```rust,ignore
/// use motlie_db::{Writer, AddNode, MutationRunnable, Id, TimestampMilli};
///
/// let (writer, receiver) = create_mutation_writer(Default::default());
///
/// // Send a single mutation using .run() pattern
/// AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     valid_range: None,
/// }
/// .run(&writer)
/// .await?;
/// ```
///
/// ## Batch Mutations
///
/// ```rust,ignore
/// use motlie_db::{mutations, AddNode, AddEdge, MutationRunnable};
///
/// // Send multiple mutations in a batch
/// mutations![
///     AddNode { /* ... */ },
///     AddEdge { /* ... */ },
/// ]
/// .run(&writer)
/// .await?;
/// ```
///
/// ## Transactions (Read-Your-Writes)
///
/// ```rust,ignore
/// let mut txn = writer.transaction()?;
///
/// txn.write(AddNode { id, ... })?;
/// let result = txn.read(NodeById::new(id, None))?;  // Sees uncommitted AddNode!
/// txn.write(AddEdge { ... })?;
///
/// txn.commit()?;  // Atomic commit
/// ```
///
/// See [Mutation API Guide](../docs/mutation-api-guide.md) for complete documentation.
#[derive(Clone)]
pub struct Writer {
    sender: mpsc::Sender<MutationRequest>,
    /// Processor for creating transactions (optional - only present when configured)
    /// (claude, 2026-02-07, FIXED: P2.2 - Writer holds Arc<Processor> instead of Storage)
    processor: Option<Arc<GraphProcessor>>,
    /// Optional sender for transaction mutation forwarding (e.g., to fulltext)
    transaction_forward_to: Option<mpsc::Sender<MutationRequest>>,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("sender", &"<mpsc::Sender>")
            .field("processor", &self.processor.as_ref().map(|_| "<Arc<Processor>>"))
            .field(
                "transaction_forward_to",
                &self.transaction_forward_to.as_ref().map(|_| "<mpsc::Sender>"),
            )
            .finish()
    }
}

impl Writer {
    /// Create a new MutationWriter with the given sender.
    ///
    /// This creates a basic writer without transaction support.
    /// Use `with_processor()` to enable transaction features.
    pub fn new(sender: mpsc::Sender<MutationRequest>) -> Self {
        Writer {
            sender,
            processor: None,
            transaction_forward_to: None,
        }
    }

    /// Create a new MutationWriter with processor for transaction support.
    /// (claude, 2026-02-07, FIXED: P2.2 - Primary construction with Processor)
    pub(crate) fn with_processor(sender: mpsc::Sender<MutationRequest>, processor: Arc<GraphProcessor>) -> Self {
        Writer {
            sender,
            processor: Some(processor),
            transaction_forward_to: None,
        }
    }

    /// Set the processor for transaction support.
    ///
    /// This enables transaction creation via `Writer::transaction()`.
    /// The processor provides access to the underlying storage for
    /// executing mutations and queries within a transaction scope.
    // (claude, 2026-02-07, FIXED: Made public for integration test access per codex eval)
    pub fn set_processor(&mut self, processor: Arc<GraphProcessor>) {
        self.processor = Some(processor);
    }

    /// Set the sender for transaction mutation forwarding.
    ///
    /// When transactions commit, their mutations will be forwarded
    /// to this sender (best-effort, non-blocking).
    pub fn set_transaction_forward_to(&mut self, sender: mpsc::Sender<MutationRequest>) {
        self.transaction_forward_to = Some(sender);
    }

    /// Begin a transaction for read-your-writes semantics.
    ///
    /// The returned Transaction allows interleaved writes and reads
    /// within a single atomic scope. Transaction lifetime is tied to
    /// the Writer's processor/storage.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    ///
    /// txn.write(AddNode { ... })?;
    /// let result = txn.read(NodeById::new(id, None))?;  // Sees the AddNode
    /// txn.write(AddEdge { ... })?;
    ///
    /// txn.commit()?;  // Sync commit, best-effort forwarding
    /// ```
    ///
    /// # Concurrent Transactions
    ///
    /// To run concurrent transactions, clone the Writer first:
    ///
    /// ```rust,ignore
    /// let writer1 = writer.clone();
    /// let writer2 = writer.clone();
    ///
    /// // Now can create transactions on each
    /// let txn1 = writer1.transaction()?;
    /// let txn2 = writer2.transaction()?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Processor/Storage is not configured
    /// - Storage is not in read-write mode
    pub fn transaction(&self) -> Result<Transaction<'_>> {
        // (claude, 2026-02-07, FIXED: P2.2/P3.3 - Use processor for transactions)
        let processor = self
            .processor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Writer not configured with processor for transactions"))?;

        let txn_db = processor.transaction_db()?;
        let txn = txn_db.transaction();
        let name_cache = processor.name_cache().clone();

        Ok(Transaction::new(txn, txn_db, self.transaction_forward_to.clone(), name_cache))
    }

    /// Check if transactions are supported by this writer.
    pub fn supports_transactions(&self) -> bool {
        // Check processor's storage if available
        if let Some(p) = &self.processor {
            return p.storage().is_transactional();
        }
        false
    }

    /// Send a batch of mutations to be processed asynchronously.
    ///
    /// This method returns immediately after enqueueing the mutations.
    /// Use `flush()` to wait for mutations to be committed, or use
    /// `send_sync()` to send and wait in one call.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Fire-and-forget (high throughput)
    /// writer.send(vec![AddNode { ... }.into()]).await?;
    /// writer.send(vec![AddEdge { ... }.into()]).await?;
    ///
    /// // Wait for all to be committed
    /// writer.flush().await?;
    /// ```
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.sender
            .send(MutationRequest {
                payload: mutations,
                options: ExecOptions::default(),
                reply: None,
                timeout: None,
                request_id: new_request_id(),
                created_at: std::time::Instant::now(),
            })
            .await
            .context("Failed to send mutations to writer queue")
    }

    /// Send mutations with execution options and wait for a reply.
    ///
    /// Returns a per-mutation reply vector in the same order as the input.
    pub async fn send_with_result(
        &self,
        mutations: Vec<Mutation>,
        options: ExecOptions,
    ) -> Result<ReplyEnvelope<Vec<MutationResult>>> {
        if mutations.is_empty() {
            return Ok(ReplyEnvelope::new(new_request_id(), 0, Vec::new()));
        }

        let (tx, rx) = oneshot::channel();
        self.sender
            .send(MutationRequest {
                payload: mutations,
                options,
                reply: Some(tx),
                timeout: None,
                request_id: new_request_id(),
                created_at: std::time::Instant::now(),
            })
            .await
            .context("Failed to send mutations to writer queue")?;

        rx.await.context("Mutation reply channel dropped")?
    }

    /// Flush all pending mutations and wait for commit.
    ///
    /// Returns when all mutations sent before this call are committed
    /// to RocksDB and visible to readers.
    ///
    /// # Fulltext Consistency
    ///
    /// This method only guarantees **graph (RocksDB) consistency**.
    /// Fulltext indexing continues asynchronously and may not be complete
    /// when this method returns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send mutations
    /// writer.send(vec![AddNode { ... }.into()]).await?;
    /// writer.send(vec![AddEdge { ... }.into()]).await?;
    ///
    /// // Wait for all to be committed
    /// writer.flush().await?;
    ///
    /// // Now safe to read
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The writer channel is closed
    /// - The consumer task has panicked or been dropped
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        // Send flush marker through the same channel as mutations
        self.sender
            .send(MutationRequest {
                payload: vec![Mutation::Flush(FlushMarker::new(tx))],
                options: ExecOptions::default(),
                reply: None,
                timeout: None,
                request_id: new_request_id(),
                created_at: std::time::Instant::now(),
            })
            .await
            .context("Failed to send flush marker - channel closed")?;

        // Wait for consumer to process it
        rx.await
            .context("Flush failed - consumer dropped completion channel")?;

        Ok(())
    }

    /// Send mutations and wait for commit.
    ///
    /// This is a convenience method equivalent to `send()` followed by `flush()`.
    /// Returns when all mutations are visible to readers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send and wait in one call
    /// writer.send_sync(vec![
    ///     AddNode { ... }.into(),
    ///     AddEdge { ... }.into(),
    /// ]).await?;
    ///
    /// // Immediately visible
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }
        self.send(mutations).await?;
        self.flush().await
    }

    /// Check if the writer is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

/// Create a new mutation writer and receiver pair
pub fn create_mutation_writer(config: WriterConfig) -> (Writer, mpsc::Receiver<MutationRequest>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::new(sender);
    (writer, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Generic consumer that processes mutations using a Processor
pub struct Consumer<P: Processor> {
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    processor: P,
    /// Optional sender to forward mutations to the next consumer in the chain
    next: Option<mpsc::Sender<MutationRequest>>,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    pub fn new(
        receiver: mpsc::Receiver<MutationRequest>,
        config: WriterConfig,
        processor: P,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: None,
        }
    }

    /// Create a new Consumer that forwards mutations to the next consumer in the chain
    pub fn with_next(
        receiver: mpsc::Receiver<MutationRequest>,
        config: WriterConfig,
        processor: P,
        next: mpsc::Sender<MutationRequest>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: Some(next),
        }
    }

    /// Process mutations continuously until the channel is closed
    #[tracing::instrument(skip(self), name = "mutation_consumer")]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting mutation consumer");

        loop {
            // Wait for the next batch of mutations
            match self.receiver.recv().await {
                Some(request) => {
                    // Process the batch immediately
                    self.process_request(request).await?;
                }
                None => {
                    // Channel closed
                    tracing::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations
    #[tracing::instrument(skip(self, request), fields(batch_size = request.payload.len()))]
    async fn process_request(&self, request: MutationRequest) -> Result<()> {
        let request_id = request.request_id;
        let elapsed = request.elapsed_nanos();
        let MutationRequest {
            payload,
            options,
            reply,
            ..
        } = request;
        let mutations = payload;

        // Log what we're processing
        for mutation in &mutations {
            match mutation {
                Mutation::AddNode(args) => {
                    tracing::debug!(id = %args.id, name = %args.name, "Processing AddNode");
                }
                Mutation::AddEdge(args) => {
                    tracing::debug!(
                        source = %args.source_node_id,
                        target = %args.target_node_id,
                        name = %args.name,
                        "Processing AddEdge"
                    );
                }
                Mutation::AddNodeFragment(args) => {
                    tracing::debug!(
                        id = %args.id,
                        body_len = args.content.0.len(),
                        "Processing AddNodeFragment"
                    );
                }
                Mutation::AddEdgeFragment(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.edge_name,
                        body_len = args.content.0.len(),
                        "Processing AddEdgeFragment"
                    );
                }
                // CONTENT-ADDRESS: Update/Delete mutations
                Mutation::UpdateNode(args) => {
                    tracing::debug!(
                        id = %args.id,
                        expected_version = args.expected_version,
                        has_active_period = args.new_active_period.is_some(),
                        has_summary = args.new_summary.is_some(),
                        "Processing UpdateNode"
                    );
                }
                Mutation::UpdateEdge(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.name,
                        expected_version = args.expected_version,
                        has_weight = args.new_weight.is_some(),
                        has_active_period = args.new_active_period.is_some(),
                        has_summary = args.new_summary.is_some(),
                        "Processing UpdateEdge"
                    );
                }
                Mutation::DeleteNode(args) => {
                    tracing::debug!(
                        id = %args.id,
                        expected_version = args.expected_version,
                        "Processing DeleteNode"
                    );
                }
                Mutation::DeleteEdge(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.name,
                        expected_version = args.expected_version,
                        "Processing DeleteEdge"
                    );
                }
                // (claude, 2026-02-07, FIXED: use as_of instead of target_version - Codex Item 1)
                Mutation::RestoreNode(args) => {
                    tracing::debug!(
                        id = %args.id,
                        as_of = ?args.as_of,
                        "Processing RestoreNode"
                    );
                }
                Mutation::RestoreEdge(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.name,
                        as_of = ?args.as_of,
                        "Processing RestoreEdge"
                    );
                }
                Mutation::Flush(_) => {
                    tracing::debug!("Processing Flush marker");
                }
            }
        }

        // Process all mutations in a single call
        // (Flush markers are no-ops in storage but are included for ordering)
        let replies_result = self
            .processor
            .process_mutations_with_options(&mutations, options)
            .await
            .with_context(|| format!("Failed to process mutations: {:?}", mutations));

        let _replies = match replies_result {
            Ok(replies) => {
                let envelope = ReplyEnvelope::new(request_id, elapsed, replies);
                if let Some(sender) = reply {
                    let _ = sender.send(Ok(envelope.clone()));
                }
                envelope
            }
            Err(err) => {
                if let Some(sender) = reply {
                    let _ = sender.send(Err(anyhow::anyhow!(err.to_string())));
                }
                return Err(err);
            }
        };

        // After successful commit, signal completion for any flush markers
        // This guarantees that all mutations before the flush are now visible to readers
        for mutation in &mutations {
            if let Mutation::Flush(marker) = mutation {
                if let Some(completion) = marker.take_completion() {
                    // Signal that flush is complete - ignore send errors
                    // (receiver may have been dropped if caller timed out)
                    let _ = completion.send(());
                    tracing::debug!("Flush completion signaled");
                }
            }
        }

        // Forward the batch to the next consumer in the chain if configured
        // This is a best-effort send - if the buffer is full, we log and continue
        // Note: We filter out Flush markers when forwarding - they are local to this consumer
        if let Some(sender) = &self.next {
            if options.dry_run {
                return Ok(());
            }

            let non_flush_mutations: Vec<_> = mutations
                .iter()
                .filter(|m: &&Mutation| !m.is_flush())
                .cloned()
                .collect();

            if !non_flush_mutations.is_empty() {
                if let Err(e) = sender.try_send(MutationRequest {
                    payload: non_flush_mutations,
                    options,
                    reply: None,
                    timeout: None,
                    request_id: new_request_id(),
                    created_at: std::time::Instant::now(),
                }) {
                    tracing::warn!(
                        err = %e,
                        count = mutations.len(),
                        "[BUFFER FULL] Next consumer busy, dropping mutations"
                    );
                }
            }
        }

        Ok(())
    }
}

/// Spawn a mutation consumer as a background task.
///
/// This is the generic helper used internally and by the fulltext module.
pub fn spawn_consumer<P: Processor + 'static>(
    consumer: Consumer<P>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Path-based Consumer Functions (Test convenience)
// (claude, 2026-02-07) - Convenience wrappers for tests
// ============================================================================

use std::path::Path;
use tokio::task::JoinHandle;

/// Spawn a mutation consumer with path-based storage creation.
///
/// Convenience function for tests that creates storage at the given path.
/// Uses the new Processor-based infrastructure internally.
///
/// # Arguments
/// * `receiver` - Mutation receiver from create_mutation_writer
/// * `config` - Writer configuration
/// * `db_path` - Path to create/open the database
///
/// # Returns
/// JoinHandle for the spawned consumer task
pub fn spawn_mutation_consumer(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    // Create storage at path
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to initialize storage");
    let storage = Arc::new(storage);

    // Create processor and consumer
    let processor = Arc::new(GraphProcessor::new(storage));
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
}

/// Spawn a mutation consumer with chaining to next consumer.
///
/// Convenience function for tests that creates storage at the given path
/// and forwards mutations to a next consumer in the chain.
///
/// # Arguments
/// * `receiver` - Mutation receiver from create_mutation_writer
/// * `config` - Writer configuration
/// * `db_path` - Path to create/open the database
/// * `next` - Sender for forwarding mutations to next consumer
///
/// # Returns
/// JoinHandle for the spawned consumer task
pub fn spawn_mutation_consumer_with_next(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    db_path: &Path,
    next: mpsc::Sender<MutationRequest>,
) -> JoinHandle<Result<()>> {
    // Create storage at path
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to initialize storage");
    let storage = Arc::new(storage);

    // Create processor and consumer with chaining
    let processor = Arc::new(GraphProcessor::new(storage));
    let consumer = Consumer::with_next(receiver, config, processor, next);
    spawn_consumer(consumer)
}

/// Spawn a mutation consumer with shared Processor and existing receiver.
///
/// Convenience function that matches the old `spawn_mutation_consumer_with_graph` signature.
/// Uses the Processor to process mutations from an existing receiver.
///
/// # Arguments
/// * `receiver` - Mutation receiver from create_mutation_writer
/// * `config` - Writer configuration
/// * `processor` - Shared GraphProcessor instance
///
/// # Returns
/// JoinHandle for the spawned consumer task
pub fn spawn_mutation_consumer_with_receiver(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    processor: Arc<GraphProcessor>,
) -> JoinHandle<Result<()>> {
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
}

// ============================================================================
// Processor-based Consumer Functions (ARCH2 Pattern)
// (claude, 2026-02-07, FIXED: P2.4 - Construction helpers)
// ============================================================================

/// Spawn a mutation consumer with storage.
///
/// This is the primary public construction helper.
/// Creates Processor internally and spawns consumer task.
///
/// # Arguments
/// * `storage` - Shared storage instance
/// * `config` - Writer configuration
///
/// # Returns
/// Tuple of (Writer, JoinHandle) for the spawned consumer
pub fn spawn_mutation_consumer_with_storage(
    storage: Arc<Storage>,
    config: WriterConfig,
) -> (Writer, JoinHandle<Result<()>>) {
    let processor = Arc::new(GraphProcessor::new(storage));
    spawn_mutation_consumer_with_processor(processor, config)
}

/// Spawn a mutation consumer with existing processor.
///
/// Used when Processor is shared (e.g., with Reader).
/// This is pub(crate) - use spawn_mutation_consumer_with_storage for public API.
///
/// # Arguments
/// * `processor` - Shared GraphProcessor instance
/// * `config` - Writer configuration
///
/// # Returns
/// Tuple of (Writer, JoinHandle) for the spawned consumer
pub(crate) fn spawn_mutation_consumer_with_processor(
    processor: Arc<GraphProcessor>,
    config: WriterConfig,
) -> (Writer, JoinHandle<Result<()>>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::with_processor(sender, processor.clone());

    // Create consumer with processor (implements writer::Processor trait)
    let consumer = Consumer::new(receiver, config, processor);
    let handle = spawn_consumer(consumer);

    (writer, handle)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mutation::{AddEdge, AddNode, AddNodeFragment, UpdateEdge};
    use crate::writer::Runnable as MutRunnable;
    use super::super::schema::{EdgeSummary, NodeSummary};
    use crate::{DataUrl, Id, TimestampMilli};
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_writer_closed_detection() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config);

        assert!(!writer.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Writer should detect channel is closed
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(writer.is_closed());
    }

    #[tokio::test]
    async fn test_writer_send_operations() {
        let config = WriterConfig::default();
        let (writer, _receiver) = create_mutation_writer(config);

        // Test that all send operations work with new API
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("test node summary"),
        };

        let edge_args = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
            summary: EdgeSummary::from_text("edge summary"),
            weight: Some(1.0),
            valid_range: None,
        };

        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("test fragment"),
            valid_range: None,
        };

        let src_id = Id::new();
        let dst_id = Id::new();
        let update_args = UpdateEdge {
            src_id,
            dst_id,
            name: "test_edge".to_string(),
            expected_version: 1,
            new_weight: Some(Some(0.5)),
            new_active_period: None,
            new_summary: None,
        };

        // Test new mutation API
        node_args.run(&writer).await.unwrap();
        edge_args.run(&writer).await.unwrap();
        fragment_args.run(&writer).await.unwrap();
        update_args.run(&writer).await.unwrap();
    }
}
