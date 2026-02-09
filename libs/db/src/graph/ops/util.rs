//! Utility functions for graph operations.

use std::ops::Bound;

use crate::TimestampMilli;

/// Check if a timestamp falls within a time range.
///
/// # Arguments
///
/// * `ts` - The timestamp to check
/// * `time_range` - A tuple of (start_bound, end_bound)
///
/// # Returns
///
/// `true` if the timestamp is within the range, `false` otherwise.
pub(crate) fn timestamp_in_range(
    ts: TimestampMilli,
    time_range: &(Bound<TimestampMilli>, Bound<TimestampMilli>),
) -> bool {
    let (start_bound, end_bound) = time_range;

    let start_ok = match start_bound {
        Bound::Unbounded => true,
        Bound::Included(start) => ts.0 >= start.0,
        Bound::Excluded(start) => ts.0 > start.0,
    };

    let end_ok = match end_bound {
        Bound::Unbounded => true,
        Bound::Included(end) => ts.0 <= end.0,
        Bound::Excluded(end) => ts.0 < end.0,
    };

    start_ok && end_ok
}
