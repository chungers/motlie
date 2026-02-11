use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use motlie_db::graph::scan::{
    AllEdgeFragments, AllEdgeSummaries, AllEdgeSummaryIndex, AllEdgeVersionHistory, AllEdges,
    AllGraphMeta, AllNames, AllNodeFragments, AllNodeSummaries, AllNodeSummaryIndex,
    AllNodeVersionHistory, AllNodes, AllOrphanSummaries, AllReverseEdges, EdgeFragmentRecord,
    EdgeRecord, EdgeSummaryIndexRecord, EdgeSummaryRecord, EdgeVersionHistoryRecord,
    GraphMetaRecord, NameRecord, NodeFragmentRecord, NodeRecord, NodeSummaryIndexRecord,
    NodeSummaryRecord, NodeVersionHistoryRecord, OrphanSummaryRecord, ReverseEdgeRecord, Visitable,
};
use motlie_db::graph::{NameHash, Storage, SummaryHash};
use motlie_db::{DataUrl, Id, TimestampMilli};
use std::path::PathBuf;

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(Debug, ClapArgs)]
#[clap(args_conflicts_with_subcommands = false)]
pub struct Command {
    /// Path to the RocksDB database directory
    #[clap(long, short = 'p')]
    pub db_dir: PathBuf,

    #[clap(subcommand)]
    pub verb: Verb,
}

#[derive(Debug, Subcommand)]
pub enum Verb {
    /// List column families
    List(List),
    /// Dump contents of a column family
    Scan(Scan),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ColumnFamily {
    #[value(name = "graph/nodes")]
    Nodes,
    #[value(name = "graph/node_fragments")]
    NodeFragments,
    #[value(name = "graph/edge_fragments")]
    EdgeFragments,
    #[value(name = "graph/forward_edges")]
    OutgoingEdges,
    #[value(name = "graph/reverse_edges")]
    IncomingEdges,
    #[value(name = "graph/names")]
    Names,
    #[value(name = "graph/node_summaries")]
    NodeSummaries,
    #[value(name = "graph/edge_summaries")]
    EdgeSummaries,
    #[value(name = "graph/node_summary_index")]
    NodeSummaryIndex,
    #[value(name = "graph/edge_summary_index")]
    EdgeSummaryIndex,
    #[value(name = "graph/node_version_history")]
    NodeVersionHistory,
    #[value(name = "graph/edge_version_history")]
    EdgeVersionHistory,
    #[value(name = "graph/orphan_summaries")]
    OrphanSummaries,
    #[value(name = "graph/meta")]
    GraphMeta,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Default)]
pub enum OutputFormat {
    /// Tab-separated values
    Tsv,
    /// Formatted table with aligned columns (default)
    #[default]
    Table,
}

#[derive(Debug, ClapArgs)]
pub struct List {}

#[derive(Debug, ClapArgs)]
pub struct Scan {
    /// Column family to dump
    #[clap(value_enum)]
    pub cf: ColumnFamily,

    /// Reference time for temporal validity check.
    /// Only records valid at this time will be returned.
    /// Format: YYYY-MM-DD or YYYY-MM-DD-HH:mm:ss
    /// If not specified, all records are returned regardless of temporal validity.
    #[clap(value_name = "DATETIME")]
    pub at: Option<String>,

    /// Last ID from previous page (exclusive start for pagination)
    #[clap(long)]
    pub last: Option<String>,

    /// Maximum number of results (default: 100)
    #[clap(long, default_value = "100")]
    pub limit: usize,

    /// Output format
    #[clap(long, short = 'f', value_enum, default_value = "table")]
    pub format: OutputFormat,

    /// Scan in reverse direction (from end to start)
    #[clap(long, short = 'r')]
    pub reverse: bool,
}

pub fn run(cmd: &Command) {
    trace!("Running command: {:?}", cmd);

    match &cmd.verb {
        Verb::List(_args) => {
            run_list();
        }
        Verb::Scan(args) => {
            if let Err(e) = run_scan(&cmd.db_dir, args) {
                error!("Dump failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_list() {
    println!("Column families:");
    println!("  graph/names");
    println!("  graph/nodes");
    println!("  graph/node_fragments");
    println!("  graph/node_summaries");
    println!("  graph/node_summary_index");
    println!("  graph/node_version_history");
    println!("  graph/edge_fragments");
    println!("  graph/edge_summaries");
    println!("  graph/edge_summary_index");
    println!("  graph/edge_version_history");
    println!("  graph/forward_edges");
    println!("  graph/reverse_edges");
    println!("  graph/orphan_summaries");
    println!("  graph/meta");
}

fn run_scan(db_dir: &PathBuf, args: &Scan) -> anyhow::Result<()> {
    let mut storage = Storage::readonly(db_dir);
    storage.ready()?;

    // Parse reference timestamp if provided
    let reference_ts = match &args.at {
        Some(datetime_str) => Some(parse_datetime(datetime_str)?),
        None => None,
    };

    match args.cf {
        ColumnFamily::Nodes => scan_nodes(&storage, args, reference_ts),
        ColumnFamily::NodeFragments => scan_node_fragments(&storage, args, reference_ts),
        ColumnFamily::EdgeFragments => scan_edge_fragments(&storage, args, reference_ts),
        ColumnFamily::OutgoingEdges => scan_outgoing_edges(&storage, args, reference_ts),
        ColumnFamily::IncomingEdges => scan_incoming_edges(&storage, args, reference_ts),
        ColumnFamily::Names => scan_names(&storage, args),
        ColumnFamily::NodeSummaries => scan_node_summaries(&storage, args),
        ColumnFamily::EdgeSummaries => scan_edge_summaries(&storage, args),
        ColumnFamily::NodeSummaryIndex => scan_node_summary_index(&storage, args),
        ColumnFamily::EdgeSummaryIndex => scan_edge_summary_index(&storage, args),
        ColumnFamily::NodeVersionHistory => scan_node_version_history(&storage, args),
        ColumnFamily::EdgeVersionHistory => scan_edge_version_history(&storage, args),
        ColumnFamily::OrphanSummaries => scan_orphan_summaries(&storage, args),
        ColumnFamily::GraphMeta => scan_graph_meta(&storage, args),
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a datetime string to TimestampMilli.
/// Supports formats:
/// - YYYY-MM-DD (assumes 00:00:00 time)
/// - YYYY-MM-DD-HH:mm:ss
fn parse_datetime(s: &str) -> anyhow::Result<TimestampMilli> {
    // Try parsing as YYYY-MM-DD-HH:mm:ss first
    if let Some((date_part, time_part)) = s.split_once('-').and_then(|_| {
        // Split on the 4th hyphen (after YYYY-MM-DD-)
        let parts: Vec<&str> = s.splitn(4, '-').collect();
        if parts.len() == 4 {
            Some((
                format!("{}-{}-{}", parts[0], parts[1], parts[2]),
                parts[3].to_string(),
            ))
        } else {
            None
        }
    }) {
        // Parse date: YYYY-MM-DD
        let date_parts: Vec<u32> = date_part
            .split('-')
            .map(|p| p.parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Invalid date format '{}': {}", date_part, e))?;

        if date_parts.len() != 3 {
            anyhow::bail!("Invalid date format '{}': expected YYYY-MM-DD", date_part);
        }

        let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);

        // Parse time: HH:mm:ss
        let time_parts: Vec<u32> = time_part
            .split(':')
            .map(|p| p.parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("Invalid time format '{}': {}", time_part, e))?;

        if time_parts.len() != 3 {
            anyhow::bail!("Invalid time format '{}': expected HH:mm:ss", time_part);
        }

        let (hour, minute, second) = (time_parts[0], time_parts[1], time_parts[2]);

        // Convert to timestamp using a simple calculation
        // This is a simplified calculation - doesn't handle leap seconds etc.
        let ts = datetime_to_millis(year, month, day, hour, minute, second)?;
        return Ok(TimestampMilli(ts));
    }

    // Try parsing as YYYY-MM-DD (date only, assume 00:00:00)
    let parts: Vec<u32> = s
        .split('-')
        .map(|p| p.parse())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Invalid date format '{}': {}", s, e))?;

    if parts.len() != 3 {
        anyhow::bail!(
            "Invalid datetime format '{}': expected YYYY-MM-DD or YYYY-MM-DD-HH:mm:ss",
            s
        );
    }

    let (year, month, day) = (parts[0], parts[1], parts[2]);
    let ts = datetime_to_millis(year, month, day, 0, 0, 0)?;
    Ok(TimestampMilli(ts))
}

/// Convert date/time components to milliseconds since Unix epoch.
fn datetime_to_millis(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> anyhow::Result<u64> {
    // Validate ranges
    if !(1970..=9999).contains(&year) {
        anyhow::bail!("Year must be between 1970 and 9999");
    }
    if !(1..=12).contains(&month) {
        anyhow::bail!("Month must be between 1 and 12");
    }
    if !(1..=31).contains(&day) {
        anyhow::bail!("Day must be between 1 and 31");
    }
    if hour > 23 {
        anyhow::bail!("Hour must be between 0 and 23");
    }
    if minute > 59 {
        anyhow::bail!("Minute must be between 0 and 59");
    }
    if second > 59 {
        anyhow::bail!("Second must be between 0 and 59");
    }

    // Days in each month (non-leap year)
    let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    fn is_leap_year(y: u32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
    }

    // Count days from 1970-01-01 to the given date
    let mut total_days: u64 = 0;

    // Add days for complete years
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in the current year
    for m in 1..month {
        total_days += days_in_month[(m - 1) as usize] as u64;
        if m == 2 && is_leap_year(year) {
            total_days += 1; // February in leap year
        }
    }

    // Add remaining days (day is 1-indexed, so subtract 1)
    total_days += (day - 1) as u64;

    // Convert to milliseconds
    let total_seconds =
        total_days * 86400 + hour as u64 * 3600 + minute as u64 * 60 + second as u64;
    Ok(total_seconds * 1000)
}

/// Format a temporal range's start time as a string
fn format_since(valid_range: &Option<motlie_db::ActivePeriod>, format: OutputFormat) -> String {
    match valid_range {
        None => "-".to_string(),
        Some(range) => match range.0 {
            None => "-".to_string(),
            Some(ts) => format_timestamp(ts.0, format),
        },
    }
}

/// Format a temporal range's end time as a string
fn format_until(valid_range: &Option<motlie_db::ActivePeriod>, format: OutputFormat) -> String {
    match valid_range {
        None => "-".to_string(),
        Some(range) => match range.1 {
            None => "-".to_string(),
            Some(ts) => format_timestamp(ts.0, format),
        },
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

// ============================================================================
// Scan Functions
// ============================================================================

fn scan_nodes(
    storage: &Storage,
    args: &Scan,
    reference_ts: Option<TimestampMilli>,
) -> anyhow::Result<()> {
    let last = args
        .last
        .as_ref()
        .map(|s| Id::from_str(s))
        .transpose()
        .map_err(|e| anyhow::anyhow!("Invalid ID '{}': {}", args.last.as_ref().unwrap(), e))?;

    let scan = AllNodes {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "ID", "NAME"], args.format);

    scan.accept(storage, &mut |record: &NodeRecord| {
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.id.to_string(),
            record.name.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_node_fragments(
    storage: &Storage,
    args: &Scan,
    reference_ts: Option<TimestampMilli>,
) -> anyhow::Result<()> {
    // For node fragments, last format is "node_id:timestamp"
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'node_id:timestamp', got '{}'",
                s
            );
        }
        let node_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid node ID '{}': {}", parts[0], e))?;
        let timestamp = parts[1]
            .parse::<u64>()
            .map_err(|e| anyhow::anyhow!("Invalid timestamp '{}': {}", parts[1], e))?;
        Some((node_id, motlie_db::TimestampMilli(timestamp)))
    } else {
        None
    };

    let scan = AllNodeFragments {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(
        vec!["SINCE", "UNTIL", "NODE_ID", "TIMESTAMP", "MIME", "CONTENT"],
        args.format,
    );

    scan.accept(storage, &mut |record: &NodeFragmentRecord| {
        let mime = record
            .content
            .mime_type()
            .unwrap_or_else(|_| "unknown".to_string());
        let content_preview = extract_printable_content(&record.content, 60);
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.node_id.to_string(),
            record.timestamp.0.to_string(),
            mime,
            content_preview,
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_edge_fragments(
    storage: &Storage,
    args: &Scan,
    reference_ts: Option<TimestampMilli>,
) -> anyhow::Result<()> {
    // For edge fragments, last format is "src_id:dst_id:edge_name:timestamp"
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(4, ':').collect();
        if parts.len() != 4 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'src_id:dst_id:edge_name:timestamp', got '{}'",
                s
            );
        }
        let src_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[0], e))?;
        let dst_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[1], e))?;
        let edge_name = parts[2].to_string();
        let timestamp = parts[3]
            .parse::<u64>()
            .map_err(|e| anyhow::anyhow!("Invalid timestamp '{}': {}", parts[3], e))?;
        Some((
            src_id,
            dst_id,
            edge_name,
            motlie_db::TimestampMilli(timestamp),
        ))
    } else {
        None
    };

    let scan = AllEdgeFragments {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

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
        args.format,
    );

    scan.accept(storage, &mut |record: &EdgeFragmentRecord| {
        let mime = record
            .content
            .mime_type()
            .unwrap_or_else(|_| "unknown".to_string());
        let content_preview = extract_printable_content(&record.content, 60);
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.src_id.to_string(),
            record.dst_id.to_string(),
            record.timestamp.0.to_string(),
            record.edge_name.clone(),
            mime,
            content_preview,
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_outgoing_edges(
    storage: &Storage,
    args: &Scan,
    reference_ts: Option<TimestampMilli>,
) -> anyhow::Result<()> {
    // For outgoing edges, last format is "src_id:dst_id:edge_name"
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'src_id:dst_id:edge_name', got '{}'",
                s
            );
        }
        let src_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[0], e))?;
        let dst_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[1], e))?;
        let edge_name = parts[2].to_string();
        Some((src_id, dst_id, edge_name))
    } else {
        None
    };

    let scan = AllEdges {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(
        vec!["SINCE", "UNTIL", "SRC_ID", "DST_ID", "EDGE_NAME", "WEIGHT"],
        args.format,
    );

    scan.accept(storage, &mut |record: &EdgeRecord| {
        let weight_str = record
            .weight
            .map(|w| format!("{:.4}", w))
            .unwrap_or_else(|| "-".to_string());
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.src_id.to_string(),
            record.dst_id.to_string(),
            record.name.clone(),
            weight_str,
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_incoming_edges(
    storage: &Storage,
    args: &Scan,
    reference_ts: Option<TimestampMilli>,
) -> anyhow::Result<()> {
    // For incoming edges, last format is "dst_id:src_id:edge_name"
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'dst_id:src_id:edge_name', got '{}'",
                s
            );
        }
        let dst_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[0], e))?;
        let src_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[1], e))?;
        let edge_name = parts[2].to_string();
        Some((dst_id, src_id, edge_name))
    } else {
        None
    };

    let scan = AllReverseEdges {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(
        vec!["SINCE", "UNTIL", "DST_ID", "SRC_ID", "EDGE_NAME"],
        args.format,
    );

    scan.accept(storage, &mut |record: &ReverseEdgeRecord| {
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.dst_id.to_string(),
            record.src_id.to_string(),
            record.name.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

// ============================================================================
// Hex Helpers
// ============================================================================

/// Parse a 16-char hex string to an 8-byte array (for NameHash / SummaryHash).
fn parse_hex_8(s: &str) -> anyhow::Result<[u8; 8]> {
    if s.len() != 16 {
        anyhow::bail!("Expected 16-char hex string, got '{}'", s);
    }
    let mut bytes = [0u8; 8];
    for i in 0..8 {
        bytes[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|e| anyhow::anyhow!("Invalid hex '{}': {}", s, e))?;
    }
    Ok(bytes)
}

// ============================================================================
// New Scan Functions
// ============================================================================

fn scan_names(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = args
        .last
        .as_ref()
        .map(|s| {
            let bytes = parse_hex_8(s)?;
            Ok::<_, anyhow::Error>(NameHash::from_bytes(bytes))
        })
        .transpose()?;

    let scan = AllNames {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(vec!["HASH", "NAME"], args.format);

    scan.accept(storage, &mut |record: &NameRecord| {
        printer.add_row(vec![record.hash.clone(), record.name.clone()]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_node_summaries(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = args
        .last
        .as_ref()
        .map(|s| {
            let bytes = parse_hex_8(s)?;
            Ok::<_, anyhow::Error>(SummaryHash::from_bytes(bytes))
        })
        .transpose()?;

    let scan = AllNodeSummaries {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(vec!["HASH", "MIME", "CONTENT"], args.format);

    scan.accept(storage, &mut |record: &NodeSummaryRecord| {
        let mime = record
            .content
            .mime_type()
            .unwrap_or_else(|_| "unknown".to_string());
        let content_preview = extract_printable_content(&record.content, 60);
        printer.add_row(vec![record.hash.clone(), mime, content_preview]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_edge_summaries(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = args
        .last
        .as_ref()
        .map(|s| {
            let bytes = parse_hex_8(s)?;
            Ok::<_, anyhow::Error>(SummaryHash::from_bytes(bytes))
        })
        .transpose()?;

    let scan = AllEdgeSummaries {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(vec!["HASH", "MIME", "CONTENT"], args.format);

    scan.accept(storage, &mut |record: &EdgeSummaryRecord| {
        let mime = record
            .content
            .mime_type()
            .unwrap_or_else(|_| "unknown".to_string());
        let content_preview = extract_printable_content(&record.content, 60);
        printer.add_row(vec![record.hash.clone(), mime, content_preview]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_node_summary_index(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'hash_hex:node_id:version', got '{}'",
                s
            );
        }
        let hash_bytes = parse_hex_8(parts[0])?;
        let node_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid node ID '{}': {}", parts[1], e))?;
        let version: u32 = parts[2]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid version '{}': {}", parts[2], e))?;
        Some((SummaryHash::from_bytes(hash_bytes), node_id, version))
    } else {
        None
    };

    let scan = AllNodeSummaryIndex {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(
        vec!["HASH", "NODE_ID", "VERSION", "STATUS"],
        args.format,
    );

    scan.accept(storage, &mut |record: &NodeSummaryIndexRecord| {
        printer.add_row(vec![
            record.hash.clone(),
            record.node_id.to_string(),
            record.version.to_string(),
            record.status.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_edge_summary_index(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(5, ':').collect();
        if parts.len() != 5 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'hash_hex:src_id:dst_id:name_hash_hex:version', got '{}'",
                s
            );
        }
        let hash_bytes = parse_hex_8(parts[0])?;
        let src_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[1], e))?;
        let dst_id = Id::from_str(parts[2])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[2], e))?;
        let name_hash_bytes = parse_hex_8(parts[3])?;
        let version: u32 = parts[4]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid version '{}': {}", parts[4], e))?;
        Some((
            SummaryHash::from_bytes(hash_bytes),
            src_id,
            dst_id,
            NameHash::from_bytes(name_hash_bytes),
            version,
        ))
    } else {
        None
    };

    let scan = AllEdgeSummaryIndex {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(
        vec!["HASH", "SRC_ID", "DST_ID", "EDGE_NAME", "VERSION", "STATUS"],
        args.format,
    );

    scan.accept(storage, &mut |record: &EdgeSummaryIndexRecord| {
        printer.add_row(vec![
            record.hash.clone(),
            record.src_id.to_string(),
            record.dst_id.to_string(),
            record.edge_name.clone(),
            record.version.to_string(),
            record.status.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_node_version_history(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'node_id:valid_since:version', got '{}'",
                s
            );
        }
        let node_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid node ID '{}': {}", parts[0], e))?;
        let valid_since: u64 = parts[1]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid timestamp '{}': {}", parts[1], e))?;
        let version: u32 = parts[2]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid version '{}': {}", parts[2], e))?;
        Some((node_id, TimestampMilli(valid_since), version))
    } else {
        None
    };

    let scan = AllNodeVersionHistory {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(
        vec![
            "NODE_ID", "VALID_SINCE", "VERSION", "UPDATED_AT", "SUMMARY_HASH", "NAME", "SINCE",
            "UNTIL",
        ],
        args.format,
    );

    scan.accept(storage, &mut |record: &NodeVersionHistoryRecord| {
        printer.add_row(vec![
            record.node_id.to_string(),
            format_timestamp(record.valid_since.0, args.format),
            record.version.to_string(),
            format_timestamp(record.updated_at.0, args.format),
            record.summary_hash.clone().unwrap_or_else(|| "-".to_string()),
            record.name.clone(),
            format_since(&record.active_period, args.format),
            format_until(&record.active_period, args.format),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_edge_version_history(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(5, ':').collect();
        if parts.len() != 5 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'src_id:dst_id:name_hash_hex:valid_since:version', got '{}'",
                s
            );
        }
        let src_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[0], e))?;
        let dst_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[1], e))?;
        let name_hash_bytes = parse_hex_8(parts[2])?;
        let valid_since: u64 = parts[3]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid timestamp '{}': {}", parts[3], e))?;
        let version: u32 = parts[4]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid version '{}': {}", parts[4], e))?;
        Some((
            src_id,
            dst_id,
            NameHash::from_bytes(name_hash_bytes),
            TimestampMilli(valid_since),
            version,
        ))
    } else {
        None
    };

    let scan = AllEdgeVersionHistory {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(
        vec![
            "SRC_ID",
            "DST_ID",
            "EDGE_NAME",
            "VALID_SINCE",
            "VERSION",
            "UPDATED_AT",
            "SUMMARY_HASH",
            "WEIGHT",
            "SINCE",
            "UNTIL",
        ],
        args.format,
    );

    scan.accept(storage, &mut |record: &EdgeVersionHistoryRecord| {
        let weight_str = record
            .weight
            .map(|w| format!("{:.4}", w))
            .unwrap_or_else(|| "-".to_string());
        printer.add_row(vec![
            record.src_id.to_string(),
            record.dst_id.to_string(),
            record.edge_name.clone(),
            format_timestamp(record.valid_since.0, args.format),
            record.version.to_string(),
            format_timestamp(record.updated_at.0, args.format),
            record.summary_hash.clone().unwrap_or_else(|| "-".to_string()),
            weight_str,
            format_since(&record.active_period, args.format),
            format_until(&record.active_period, args.format),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_orphan_summaries(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'timestamp:hash_hex', got '{}'",
                s
            );
        }
        let timestamp: u64 = parts[0]
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid timestamp '{}': {}", parts[0], e))?;
        let hash_bytes = parse_hex_8(parts[1])?;
        Some((TimestampMilli(timestamp), SummaryHash::from_bytes(hash_bytes)))
    } else {
        None
    };

    let scan = AllOrphanSummaries {
        last,
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(vec!["ORPHANED_AT", "HASH", "KIND"], args.format);

    scan.accept(storage, &mut |record: &OrphanSummaryRecord| {
        printer.add_row(vec![
            format_timestamp(record.orphaned_at.0, args.format),
            record.hash.clone(),
            record.kind.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn scan_graph_meta(storage: &Storage, args: &Scan) -> anyhow::Result<()> {
    let scan = AllGraphMeta {
        limit: args.limit,
        reverse: args.reverse,
    };

    let mut printer = TablePrinter::new(vec!["FIELD", "CURSOR_BYTES"], args.format);

    scan.accept(storage, &mut |record: &GraphMetaRecord| {
        printer.add_row(vec![
            record.field.clone(),
            record.cursor_bytes_hex.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

