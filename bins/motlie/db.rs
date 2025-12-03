use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use motlie_db::scan::{
    AllEdgeFragments, AllEdgeNames, AllEdges, AllNodeFragments, AllNodeNames, AllNodes,
    AllReverseEdges, EdgeFragmentRecord, EdgeNameRecord, EdgeRecord, NodeFragmentRecord,
    NodeNameRecord, NodeRecord, ReverseEdgeRecord, Visitable,
};
use motlie_db::{DataUrl, Id, Storage, TimestampMilli};
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
    Dump(Dump),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ColumnFamily {
    Nodes,
    NodeFragments,
    EdgeFragments,
    OutgoingEdges,
    IncomingEdges,
    NodeNames,
    EdgeNames,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Default)]
pub enum OutputFormat {
    /// Tab-separated values (default)
    #[default]
    Tsv,
    /// Formatted table with aligned columns
    Table,
}

#[derive(Debug, ClapArgs)]
pub struct List {}

#[derive(Debug, ClapArgs)]
pub struct Dump {
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
    #[clap(long, short = 'f', value_enum, default_value = "tsv")]
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
        Verb::Dump(args) => {
            if let Err(e) = run_dump(&cmd.db_dir, args) {
                error!("Dump failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_list() {
    println!("Column families:");
    println!("  nodes");
    println!("  node-fragments");
    println!("  edge-fragments");
    println!("  outgoing-edges");
    println!("  incoming-edges");
    println!("  node-names");
    println!("  edge-names");
}

fn run_dump(db_dir: &PathBuf, args: &Dump) -> anyhow::Result<()> {
    let mut storage = Storage::readonly(db_dir);
    storage.ready()?;

    // Parse reference timestamp if provided
    let reference_ts = match &args.at {
        Some(datetime_str) => Some(parse_datetime(datetime_str)?),
        None => None,
    };

    match args.cf {
        ColumnFamily::Nodes => dump_nodes(&storage, args, reference_ts),
        ColumnFamily::NodeFragments => dump_node_fragments(&storage, args, reference_ts),
        ColumnFamily::EdgeFragments => dump_edge_fragments(&storage, args, reference_ts),
        ColumnFamily::OutgoingEdges => dump_outgoing_edges(&storage, args, reference_ts),
        ColumnFamily::IncomingEdges => dump_incoming_edges(&storage, args, reference_ts),
        ColumnFamily::NodeNames => dump_node_names(&storage, args, reference_ts),
        ColumnFamily::EdgeNames => dump_edge_names(&storage, args, reference_ts),
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
            Some((format!("{}-{}-{}", parts[0], parts[1], parts[2]), parts[3].to_string()))
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
    let total_seconds = total_days * 86400 + hour as u64 * 3600 + minute as u64 * 60 + second as u64;
    Ok(total_seconds * 1000)
}

/// Format a temporal range's start time as a string
fn format_since(valid_range: &Option<motlie_db::scan::TemporalRange>, format: OutputFormat) -> String {
    match valid_range {
        None => "-".to_string(),
        Some(range) => match range.0 {
            None => "-".to_string(),
            Some(ts) => format_timestamp(ts.0, format),
        },
    }
}

/// Format a temporal range's end time as a string
fn format_until(valid_range: &Option<motlie_db::scan::TemporalRange>, format: OutputFormat) -> String {
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
            let cleaned: String = s.chars().take(max_len).map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).collect();
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
                .map(|(i, cell)| format!("{:width$}", cell, width = widths.get(i).copied().unwrap_or(0)))
                .collect();
            println!("{}", line.join("  "));
        }
    }
}

// ============================================================================
// Dump Functions
// ============================================================================

fn dump_nodes(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
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

fn dump_node_fragments(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
    // For node fragments, last format is "node_id:timestamp"
    let last = if let Some(s) = &args.last {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid cursor format. Expected 'node_id:timestamp', got '{}'", s);
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

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "NODE_ID", "TIMESTAMP", "MIME", "CONTENT"], args.format);

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

fn dump_edge_fragments(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
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
        vec!["SINCE", "UNTIL", "SRC_ID", "DST_ID", "TIMESTAMP", "EDGE_NAME", "MIME", "CONTENT"],
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

fn dump_outgoing_edges(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
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

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "SRC_ID", "DST_ID", "EDGE_NAME", "WEIGHT"], args.format);

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

fn dump_incoming_edges(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
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

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "DST_ID", "SRC_ID", "EDGE_NAME"], args.format);

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

fn dump_node_names(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
    // For node names, last format is "name:node_id"
    let last = if let Some(s) = &args.last {
        // Find the last ':' to split name and id (name may contain ':')
        let last_colon = s.rfind(':').ok_or_else(|| {
            anyhow::anyhow!("Invalid cursor format. Expected 'name:node_id', got '{}'", s)
        })?;
        let name = s[..last_colon].to_string();
        let node_id = Id::from_str(&s[last_colon + 1..])
            .map_err(|e| anyhow::anyhow!("Invalid node ID '{}': {}", &s[last_colon + 1..], e))?;
        Some((name, node_id))
    } else {
        None
    };

    let scan = AllNodeNames {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "NODE_ID", "NAME"], args.format);

    scan.accept(storage, &mut |record: &NodeNameRecord| {
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.node_id.to_string(),
            record.name.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}

fn dump_edge_names(storage: &Storage, args: &Dump, reference_ts: Option<TimestampMilli>) -> anyhow::Result<()> {
    // For edge names, last format is "name:src_id:dst_id"
    let last = if let Some(s) = &args.last {
        // Split from the end - last two ':' separate the IDs
        let parts: Vec<&str> = s.rsplitn(3, ':').collect();
        if parts.len() != 3 {
            anyhow::bail!(
                "Invalid cursor format. Expected 'name:src_id:dst_id', got '{}'",
                s
            );
        }
        // parts are in reverse order: [dst_id, src_id, name]
        let dst_id = Id::from_str(parts[0])
            .map_err(|e| anyhow::anyhow!("Invalid dst ID '{}': {}", parts[0], e))?;
        let src_id = Id::from_str(parts[1])
            .map_err(|e| anyhow::anyhow!("Invalid src ID '{}': {}", parts[1], e))?;
        let name = parts[2].to_string();
        Some((name, src_id, dst_id))
    } else {
        None
    };

    let scan = AllEdgeNames {
        last,
        limit: args.limit,
        reverse: args.reverse,
        reference_ts_millis: reference_ts,
    };

    let mut printer = TablePrinter::new(vec!["SINCE", "UNTIL", "SRC_ID", "DST_ID", "NAME"], args.format);

    scan.accept(storage, &mut |record: &EdgeNameRecord| {
        printer.add_row(vec![
            format_since(&record.valid_range, args.format),
            format_until(&record.valid_range, args.format),
            record.src_id.to_string(),
            record.dst_id.to_string(),
            record.name.clone(),
        ]);
        true
    })?;

    printer.print();
    Ok(())
}
