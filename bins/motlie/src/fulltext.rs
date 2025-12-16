use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use motlie_db::fulltext::{
    EdgeHit, Edges as FulltextEdgesQuery, FacetCounts, Facets as FulltextFacetsQuery,
    FuzzyLevel as DbFuzzyLevel, Index as FulltextIndex, NodeHit, Nodes as FulltextNodesQuery,
    Storage as FulltextStorage,
};
use motlie_db::graph::mutation::{AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation};
use motlie_db::graph::writer::Processor;
use motlie_db::graph::scan::{
    AllEdgeFragments, AllEdges, AllNodeFragments, AllNodes, EdgeFragmentRecord, EdgeRecord,
    NodeFragmentRecord, NodeRecord, Visitable,
};
use motlie_db::graph::Storage as GraphStorage;
use motlie_db::{DataUrl, Id, TimestampMilli};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(Debug, ClapArgs)]
#[clap(args_conflicts_with_subcommands = false)]
pub struct Command {
    /// Path to the FullText Index directory
    #[clap(long, short = 'p')]
    pub index_dir: PathBuf,

    #[clap(subcommand)]
    pub verb: Verb,
}

#[derive(Debug, Subcommand)]
pub enum Verb {
    Index(Index),
    Search(Search),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Default)]
pub enum OutputFormat {
    /// Tab-separated values
    Tsv,
    /// Formatted table with aligned columns (default)
    #[default]
    Table,
}

/// Fuzzy matching level for search queries
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Default)]
pub enum FuzzyLevel {
    /// No fuzzy matching - exact match only (default)
    #[default]
    None,
    /// Allow 1 character edit (insert, delete, substitute, or transpose)
    Low,
    /// Allow 2 character edits
    Medium,
}

impl From<FuzzyLevel> for DbFuzzyLevel {
    fn from(level: FuzzyLevel) -> Self {
        match level {
            FuzzyLevel::None => DbFuzzyLevel::None,
            FuzzyLevel::Low => DbFuzzyLevel::Low,
            FuzzyLevel::Medium => DbFuzzyLevel::Medium,
        }
    }
}

/// Index the nodes, edges, and fragments of the graph database
/// at the given directory. The index directory (--index-dir)
#[derive(Debug, ClapArgs)]
pub struct Index {
    /// Path to the graph RocksDB directory (the directory used for db commands)
    pub graph_db_dir: PathBuf,

    /// Number of records to batch before sending to the index pipeline
    #[clap(long, short = 'b', default_value = "100")]
    pub batch_size: usize,

    /// Output format
    #[clap(long, short = 'o', value_enum, default_value = "table")]
    pub format: OutputFormat,
}

#[derive(Debug, ClapArgs)]
pub struct Search {
    /// Output format
    #[clap(long, short = 'o', value_enum, default_value = "table")]
    pub format: OutputFormat,

    #[clap(subcommand)]
    pub kind: Kind,
}

#[derive(Debug, Subcommand)]
pub enum Kind {
    Nodes(Nodes),
    Edges(Edges),
    Facets(Facets),
}

#[derive(Debug, ClapArgs)]
pub struct Nodes {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    #[clap(long, short = 'l', default_value = "10")]
    pub limit: usize,

    /// Fuzzy matching level (default: None = exact match)
    #[clap(long, short = 'f', value_enum, default_value = "none")]
    pub fuzzy_level: FuzzyLevel,

    /// Filter by tags (documents must have ANY of the specified tags)
    #[clap(long, short = 't')]
    pub tags: Vec<String>,
}

#[derive(Debug, ClapArgs)]
pub struct Edges {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    #[clap(long, short = 'l', default_value = "10")]
    pub limit: usize,

    /// Fuzzy matching level (default: None = exact match)
    #[clap(long, short = 'f', value_enum, default_value = "none")]
    pub fuzzy_level: FuzzyLevel,

    /// Filter by tags (documents must have ANY of the specified tags)
    #[clap(long, short = 't')]
    pub tags: Vec<String>,
}

#[derive(Debug, ClapArgs)]
pub struct Facets {
    /// Optional filter by document types (if empty, count all)
    #[clap(long, short = 'd')]
    pub doc_type_filter: Vec<String>,

    /// Limit for number of tag facets to return
    #[clap(long, short = 'l', default_value = "50")]
    pub tags_limit: usize,
}

pub fn run(cmd: &Command) {
    trace!("Running command: {:?}", cmd);

    match &cmd.verb {
        Verb::Index(args) => {
            if let Err(e) = run_index(&cmd.index_dir, args) {
                error!("Failed to execute index command: {}", e);
            }
        }
        Verb::Search(args) => {
            if let Err(e) = run_search(&cmd.index_dir, args) {
                error!("Failed to execute search command: {}", e);
            }
        }
    }
}

fn run_index(index_dir: &PathBuf, args: &Index) -> anyhow::Result<()> {
    info!(
        "Indexing graph database at {:?} into fulltext index at {:?}",
        args.graph_db_dir, index_dir
    );

    // Check if the index directory exists and is not empty
    if index_dir.exists() {
        let is_empty = index_dir
            .read_dir()
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);
        if !is_empty {
            anyhow::bail!(
                "Index directory {:?} is not empty. Please provide an empty or non-existent directory for a new index.",
                index_dir
            );
        }
    }

    // Open the graph database in readonly mode
    let mut graph_storage = GraphStorage::readonly(&args.graph_db_dir);
    graph_storage.ready()?;

    // Create the fulltext index in readwrite mode
    let mut fulltext_storage = FulltextStorage::readwrite(index_dir);
    fulltext_storage.ready()?;
    let fulltext_index = FulltextIndex::new(Arc::new(fulltext_storage));

    // Create a tokio runtime for async operations
    let rt = Runtime::new()?;

    // Index all record types
    index_nodes(
        &rt,
        &graph_storage,
        &fulltext_index,
        args.batch_size,
        args.format,
    )?;
    index_edges(
        &rt,
        &graph_storage,
        &fulltext_index,
        args.batch_size,
        args.format,
    )?;
    index_node_fragments(
        &rt,
        &graph_storage,
        &fulltext_index,
        args.batch_size,
        args.format,
    )?;
    index_edge_fragments(
        &rt,
        &graph_storage,
        &fulltext_index,
        args.batch_size,
        args.format,
    )?;

    info!("Indexing complete");
    Ok(())
}

fn run_search(index_dir: &PathBuf, args: &Search) -> anyhow::Result<()> {
    // Check if the index directory exists
    if !index_dir.exists() {
        anyhow::bail!(
            "Index directory {:?} does not exist. Please run 'fulltext index' first.",
            index_dir
        );
    }

    // Open the fulltext index in readonly mode
    let mut fulltext_storage = FulltextStorage::readonly(index_dir);
    fulltext_storage.ready()?;

    // Create a tokio runtime for async operations
    let rt = Runtime::new()?;

    match &args.kind {
        Kind::Nodes(nodes_args) => {
            search_nodes(&rt, &fulltext_storage, nodes_args, args.format)?;
        }
        Kind::Edges(edges_args) => {
            search_edges(&rt, &fulltext_storage, edges_args, args.format)?;
        }
        Kind::Facets(facets_args) => {
            search_facets(&rt, &fulltext_storage, facets_args, args.format)?;
        }
    }

    Ok(())
}

fn search_nodes(
    rt: &Runtime,
    storage: &FulltextStorage,
    args: &Nodes,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Searching nodes for: {}", args.query);

    // Build the query
    let mut query = FulltextNodesQuery::new(args.query.clone(), args.limit);
    query = query.with_fuzzy(args.fuzzy_level.into());
    if !args.tags.is_empty() {
        query = query.with_tags(args.tags.clone());
    }

    // Execute the query
    let results: Vec<NodeHit> = rt.block_on(query.execute(storage))?;

    // Print results
    if results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    let mut printer = TablePrinter::new(vec!["SCORE", "ID", "MATCH"], format);

    for hit in &results {
        printer.add_row(vec![
            format!("{:.4}", hit.score),
            hit.id.to_string(),
            hit.match_source.to_string(),
        ]);
    }

    printer.print();
    eprintln!("Found {} results.", results.len());
    Ok(())
}

fn search_edges(
    rt: &Runtime,
    storage: &FulltextStorage,
    args: &Edges,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Searching edges for: {}", args.query);

    // Build the query
    let mut query = FulltextEdgesQuery::new(args.query.clone(), args.limit);
    query = query.with_fuzzy(args.fuzzy_level.into());
    if !args.tags.is_empty() {
        query = query.with_tags(args.tags.clone());
    }

    // Execute the query
    let results: Vec<EdgeHit> = rt.block_on(query.execute(storage))?;

    // Print results
    if results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    let mut printer = TablePrinter::new(
        vec!["SCORE", "SRC_ID", "DST_ID", "EDGE_NAME", "MATCH"],
        format,
    );

    for hit in &results {
        printer.add_row(vec![
            format!("{:.4}", hit.score),
            hit.src_id.to_string(),
            hit.dst_id.to_string(),
            hit.edge_name.clone(),
            hit.match_source.to_string(),
        ]);
    }

    printer.print();
    eprintln!("Found {} results.", results.len());
    Ok(())
}

fn search_facets(
    rt: &Runtime,
    storage: &FulltextStorage,
    args: &Facets,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Getting facet counts...");

    // Build the query
    let mut query = FulltextFacetsQuery::new();
    if !args.doc_type_filter.is_empty() {
        query = query.with_doc_type_filter(args.doc_type_filter.clone());
    }
    query = query.with_tags_limit(args.tags_limit);

    // Execute the query
    let counts: FacetCounts = rt.block_on(query.execute(storage))?;

    let mut printer = TablePrinter::new(vec!["CATEGORY", "NAME", "COUNT"], format);

    // Collect doc_types
    if !counts.doc_types.is_empty() {
        let mut doc_types: Vec<_> = counts.doc_types.iter().collect();
        doc_types.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
        for (doc_type, count) in doc_types {
            printer.add_row(vec![
                "doc_type".to_string(),
                doc_type.clone(),
                count.to_string(),
            ]);
        }
    }

    // Collect tags
    if !counts.tags.is_empty() {
        let mut tags: Vec<_> = counts.tags.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
        for (tag, count) in tags {
            printer.add_row(vec!["tag".to_string(), tag.clone(), count.to_string()]);
        }
    }

    // Collect validity
    if !counts.validity.is_empty() {
        let mut validity: Vec<_> = counts.validity.iter().collect();
        validity.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
        for (validity_type, count) in validity {
            printer.add_row(vec![
                "validity".to_string(),
                validity_type.clone(),
                count.to_string(),
            ]);
        }
    }

    printer.print();
    eprintln!(
        "Total facets: {} doc_types, {} tags, {} validity",
        counts.doc_types.len(),
        counts.tags.len(),
        counts.validity.len()
    );
    Ok(())
}

fn index_nodes(
    rt: &Runtime,
    graph_storage: &GraphStorage,
    fulltext_index: &FulltextIndex,
    batch_size: usize,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Indexing nodes...");

    let mut last_cursor: Option<Id> = None;
    let mut total_count: usize = 0;
    let mut batch_num: usize = 0;
    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "ID", "NAME"], format);

    loop {
        let scan = AllNodes {
            last: last_cursor,
            limit: batch_size,
            reverse: false,
            reference_ts_millis: None,
        };

        let mut mutations: Vec<Mutation> = Vec::with_capacity(batch_size);
        let mut last_id: Option<Id> = None;

        scan.accept(graph_storage, &mut |record: &NodeRecord| {
            // Collect record for printing
            printer.add_row(vec![
                format_since(&record.valid_range, format),
                format_until(&record.valid_range, format),
                record.id.to_string(),
                record.name.clone(),
            ]);

            // Create mutation for fulltext indexing
            let add_node = AddNode {
                id: record.id,
                ts_millis: TimestampMilli::now(),
                name: record.name.clone(),
                valid_range: record.valid_range.clone(),
                summary: record.summary.clone(),
            };
            mutations.push(Mutation::AddNode(add_node));
            last_id = Some(record.id);

            true
        })?;

        // No more records
        if mutations.is_empty() {
            break;
        }

        let count = mutations.len();
        total_count += count;
        batch_num += 1;

        // Send batch to index pipeline
        rt.block_on(fulltext_index.process_mutations(&mutations))?;
        eprintln!(
            "Indexed batch {} of nodes: {} records (total: {})",
            batch_num, count, total_count
        );

        // Update cursor for next iteration
        last_cursor = last_id;

        // If we got fewer than batch_size, we're done
        if count < batch_size {
            break;
        }
    }

    printer.print();
    info!("Indexed {} nodes", total_count);
    Ok(())
}

fn index_edges(
    rt: &Runtime,
    graph_storage: &GraphStorage,
    fulltext_index: &FulltextIndex,
    batch_size: usize,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Indexing edges...");

    let mut last_cursor: Option<(Id, Id, String)> = None;
    let mut total_count: usize = 0;
    let mut batch_num: usize = 0;
    let mut printer = TablePrinter::new(
        vec!["SINCE", "UNTIL", "SRC_ID", "DST_ID", "EDGE_NAME", "WEIGHT"],
        format,
    );

    loop {
        let scan = AllEdges {
            last: last_cursor.clone(),
            limit: batch_size,
            reverse: false,
            reference_ts_millis: None,
        };

        let mut mutations: Vec<Mutation> = Vec::with_capacity(batch_size);
        let mut last_key: Option<(Id, Id, String)> = None;

        scan.accept(graph_storage, &mut |record: &EdgeRecord| {
            let weight_str = record
                .weight
                .map(|w| format!("{:.4}", w))
                .unwrap_or_else(|| "-".to_string());

            // Collect record for printing
            printer.add_row(vec![
                format_since(&record.valid_range, format),
                format_until(&record.valid_range, format),
                record.src_id.to_string(),
                record.dst_id.to_string(),
                record.name.clone(),
                weight_str,
            ]);

            // Create mutation for fulltext indexing
            let add_edge = AddEdge {
                source_node_id: record.src_id,
                target_node_id: record.dst_id,
                ts_millis: TimestampMilli::now(),
                name: record.name.clone(),
                valid_range: record.valid_range.clone(),
                summary: record.summary.clone(),
                weight: record.weight,
            };
            mutations.push(Mutation::AddEdge(add_edge));
            last_key = Some((record.src_id, record.dst_id, record.name.clone()));

            true
        })?;

        // No more records
        if mutations.is_empty() {
            break;
        }

        let count = mutations.len();
        total_count += count;
        batch_num += 1;

        // Send batch to index pipeline
        rt.block_on(fulltext_index.process_mutations(&mutations))?;
        eprintln!(
            "Indexed batch {} of edges: {} records (total: {})",
            batch_num, count, total_count
        );

        // Update cursor for next iteration
        last_cursor = last_key;

        // If we got fewer than batch_size, we're done
        if count < batch_size {
            break;
        }
    }

    printer.print();
    info!("Indexed {} edges", total_count);
    Ok(())
}

fn index_node_fragments(
    rt: &Runtime,
    graph_storage: &GraphStorage,
    fulltext_index: &FulltextIndex,
    batch_size: usize,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Indexing node fragments...");

    let mut last_cursor: Option<(Id, TimestampMilli)> = None;
    let mut total_count: usize = 0;
    let mut batch_num: usize = 0;
    let mut printer = TablePrinter::new(
        vec!["SINCE", "UNTIL", "NODE_ID", "TIMESTAMP", "MIME", "CONTENT"],
        format,
    );

    loop {
        let scan = AllNodeFragments {
            last: last_cursor,
            limit: batch_size,
            reverse: false,
            reference_ts_millis: None,
        };

        let mut mutations: Vec<Mutation> = Vec::with_capacity(batch_size);
        let mut last_key: Option<(Id, TimestampMilli)> = None;

        scan.accept(graph_storage, &mut |record: &NodeFragmentRecord| {
            let mime = record
                .content
                .mime_type()
                .unwrap_or_else(|_| "unknown".to_string());
            let content_preview = extract_printable_content(&record.content, 60);

            // Collect record for printing
            printer.add_row(vec![
                format_since(&record.valid_range, format),
                format_until(&record.valid_range, format),
                record.node_id.to_string(),
                record.timestamp.0.to_string(),
                mime,
                content_preview,
            ]);

            // Create mutation for fulltext indexing
            let add_node_fragment = AddNodeFragment {
                id: record.node_id,
                ts_millis: record.timestamp,
                content: record.content.clone(),
                valid_range: record.valid_range.clone(),
            };
            mutations.push(Mutation::AddNodeFragment(add_node_fragment));
            last_key = Some((record.node_id, record.timestamp));

            true
        })?;

        // No more records
        if mutations.is_empty() {
            break;
        }

        let count = mutations.len();
        total_count += count;
        batch_num += 1;

        // Send batch to index pipeline
        rt.block_on(fulltext_index.process_mutations(&mutations))?;
        eprintln!(
            "Indexed batch {} of node fragments: {} records (total: {})",
            batch_num, count, total_count
        );

        // Update cursor for next iteration
        last_cursor = last_key;

        // If we got fewer than batch_size, we're done
        if count < batch_size {
            break;
        }
    }

    printer.print();
    info!("Indexed {} node fragments", total_count);
    Ok(())
}

fn index_edge_fragments(
    rt: &Runtime,
    graph_storage: &GraphStorage,
    fulltext_index: &FulltextIndex,
    batch_size: usize,
    format: OutputFormat,
) -> anyhow::Result<()> {
    info!("Indexing edge fragments...");

    let mut last_cursor: Option<(Id, Id, String, TimestampMilli)> = None;
    let mut total_count: usize = 0;
    let mut batch_num: usize = 0;
    let mut printer = TablePrinter::new(
        vec![
            "SINCE",
            "UNTIL",
            "SRC_ID",
            "DST_ID",
            "TIMESTAMP",
            "EDGE_NAME",
            "MIME",
            "CONTENT",
        ],
        format,
    );

    loop {
        let scan = AllEdgeFragments {
            last: last_cursor.clone(),
            limit: batch_size,
            reverse: false,
            reference_ts_millis: None,
        };

        let mut mutations: Vec<Mutation> = Vec::with_capacity(batch_size);
        let mut last_key: Option<(Id, Id, String, TimestampMilli)> = None;

        scan.accept(graph_storage, &mut |record: &EdgeFragmentRecord| {
            let mime = record
                .content
                .mime_type()
                .unwrap_or_else(|_| "unknown".to_string());
            let content_preview = extract_printable_content(&record.content, 60);

            // Collect record for printing
            printer.add_row(vec![
                format_since(&record.valid_range, format),
                format_until(&record.valid_range, format),
                record.src_id.to_string(),
                record.dst_id.to_string(),
                record.timestamp.0.to_string(),
                record.edge_name.clone(),
                mime,
                content_preview,
            ]);

            // Create mutation for fulltext indexing
            let add_edge_fragment = AddEdgeFragment {
                src_id: record.src_id,
                dst_id: record.dst_id,
                edge_name: record.edge_name.clone(),
                ts_millis: record.timestamp,
                content: record.content.clone(),
                valid_range: record.valid_range.clone(),
            };
            mutations.push(Mutation::AddEdgeFragment(add_edge_fragment));
            last_key = Some((
                record.src_id,
                record.dst_id,
                record.edge_name.clone(),
                record.timestamp,
            ));

            true
        })?;

        // No more records
        if mutations.is_empty() {
            break;
        }

        let count = mutations.len();
        total_count += count;
        batch_num += 1;

        // Send batch to index pipeline
        rt.block_on(fulltext_index.process_mutations(&mutations))?;
        eprintln!(
            "Indexed batch {} of edge fragments: {} records (total: {})",
            batch_num, count, total_count
        );

        // Update cursor for next iteration
        last_cursor = last_key;

        // If we got fewer than batch_size, we're done
        if count < batch_size {
            break;
        }
    }

    printer.print();
    info!("Indexed {} edge fragments", total_count);
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Simple table printer that collects rows and prints with column alignment
struct TablePrinter {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    format: OutputFormat,
}

impl TablePrinter {
    fn new(headers: Vec<&str>, format: OutputFormat) -> Self {
        Self {
            headers: headers.into_iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
            format,
        }
    }

    fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    fn print(&self) {
        match self.format {
            OutputFormat::Tsv => self.print_tsv(),
            OutputFormat::Table => self.print_table(),
        }
    }

    fn print_tsv(&self) {
        for row in &self.rows {
            println!("{}", row.join("\t"));
        }
    }

    fn print_table(&self) {
        if self.rows.is_empty() {
            return;
        }

        // Calculate column widths
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        // Print header
        let header_line: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:width$}", h, width = widths.get(i).copied().unwrap_or(0)))
            .collect();
        println!("{}", header_line.join("  "));

        // Print separator
        let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
        println!("{}", sep.join("  "));

        // Print rows
        for row in &self.rows {
            let line: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    format!(
                        "{:width$}",
                        cell,
                        width = widths.get(i).copied().unwrap_or(0)
                    )
                })
                .collect();
            println!("{}", line.join("  "));
        }
    }
}

/// Format a temporal range's start time as a string
fn format_since(valid_range: &Option<motlie_db::TemporalRange>, format: OutputFormat) -> String {
    match valid_range {
        None => format_empty_timestamp(format),
        Some(range) => match range.0 {
            None => format_empty_timestamp(format),
            Some(ts) => format_timestamp(ts.0, format),
        },
    }
}

/// Format a temporal range's end time as a string
fn format_until(valid_range: &Option<motlie_db::TemporalRange>, format: OutputFormat) -> String {
    match valid_range {
        None => format_empty_timestamp(format),
        Some(range) => match range.1 {
            None => format_empty_timestamp(format),
            Some(ts) => format_timestamp(ts.0, format),
        },
    }
}

// Fixed width for timestamp columns: YYYY-MM-DD HH:mm:ss = 19 chars
const TIMESTAMP_WIDTH: usize = 19;

/// Format an empty/missing timestamp according to the output format
fn format_empty_timestamp(format: OutputFormat) -> String {
    match format {
        OutputFormat::Tsv => "-".to_string(),
        // Fixed width to match YYYY-MM-DD HH:mm:ss (19 chars), left-aligned
        OutputFormat::Table => format!("{:<width$}", "-", width = TIMESTAMP_WIDTH),
    }
}

/// Format a timestamp according to the output format
fn format_timestamp(millis: u64, format: OutputFormat) -> String {
    match format {
        OutputFormat::Tsv => millis.to_string(),
        OutputFormat::Table => millis_to_datetime_string(millis),
    }
}

/// Convert milliseconds since epoch to YYYY-MM-DD HH:mm:ss format
fn millis_to_datetime_string(millis: u64) -> String {
    let total_seconds = millis / 1000;
    let seconds_in_day = 86400u64;
    let seconds_in_hour = 3600u64;
    let seconds_in_minute = 60u64;

    // Calculate days since epoch
    let mut remaining_days = total_seconds / seconds_in_day;
    let day_seconds = total_seconds % seconds_in_day;

    let hour = day_seconds / seconds_in_hour;
    let minute = (day_seconds % seconds_in_hour) / seconds_in_minute;
    let second = day_seconds % seconds_in_minute;

    // Calculate year, month, day from days since 1970-01-01
    let days_in_month = [31u64, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    fn is_leap_year(y: u64) -> bool {
        (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
    }

    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1u64;
    for (i, &days) in days_in_month.iter().enumerate() {
        let days_this_month = if i == 1 && is_leap_year(year) {
            days + 1
        } else {
            days
        };
        if remaining_days < days_this_month {
            break;
        }
        remaining_days -= days_this_month;
        month += 1;
    }

    let day = remaining_days + 1; // Days are 1-indexed

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hour, minute, second
    )
}

/// Check if a MIME type is printable (text-based)
fn is_printable_mime(mime: &str) -> bool {
    mime.starts_with("text/")
        || mime.starts_with("application/json")
        || mime.contains("charset=utf-8")
}

/// Extract printable content from a DataUrl, truncated to max_len
fn extract_printable_content(content: &DataUrl, max_len: usize) -> String {
    let mime = content.mime_type().unwrap_or_else(|_| String::new());
    if !is_printable_mime(&mime) {
        return String::new();
    }

    match content.decode_string() {
        Ok(s) => {
            // Replace newlines with spaces and truncate
            let cleaned: String = s
                .chars()
                .take(max_len)
                .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                .collect();
            if s.len() > max_len {
                format!("{}...", cleaned)
            } else {
                cleaned
            }
        }
        Err(_) => String::new(),
    }
}
