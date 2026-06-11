use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::time::SystemTime;

use crate::error::{DriverError, DriverResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryRecord<T> {
    pub seq: u64,
    pub at: SystemTime,
    pub item: T,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryPage<T> {
    pub items: Vec<HistoryRecord<T>>,
    pub next_after: Option<u64>,
    pub oldest_available: Option<u64>,
    pub newest_available: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HistoryBuffer<T> {
    capacity: NonZeroUsize,
    next_seq: u64,
    items: VecDeque<HistoryRecord<T>>,
    recall_index: Option<usize>,
}

impl<T> HistoryBuffer<T> {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            next_seq: 1,
            items: VecDeque::with_capacity(capacity.get()),
            recall_index: None,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity.get()
    }

    pub fn try_with_capacity(capacity: usize) -> DriverResult<Self> {
        let capacity = NonZeroUsize::new(capacity).ok_or(DriverError::InvalidHistoryCapacity)?;
        Ok(Self::new(capacity))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn latest(&self) -> Option<&HistoryRecord<T>> {
        self.items.back()
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.recall_index = None;
    }

    pub fn reset_recall(&mut self) {
        self.recall_index = None;
    }

    pub fn oldest_seq(&self) -> Option<u64> {
        self.items.front().map(|record| record.seq)
    }

    pub fn newest_seq(&self) -> Option<u64> {
        self.items.back().map(|record| record.seq)
    }

    pub fn push(&mut self, item: T) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.recall_index = None;

        if self.items.len() == self.capacity.get() {
            let _ = self.items.pop_front();
        }

        self.items.push_back(HistoryRecord {
            seq,
            at: SystemTime::now(),
            item,
        });

        seq
    }

    pub fn prev(&mut self) -> Option<&HistoryRecord<T>> {
        if self.items.is_empty() {
            self.recall_index = None;
            return None;
        }
        let index = match self.recall_index {
            Some(index) => index.saturating_sub(1),
            None => self.items.len().saturating_sub(1),
        };
        self.recall_index = Some(index);
        self.items.get(index)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&HistoryRecord<T>> {
        let index = self.recall_index?;
        let next_index = index.saturating_add(1);
        if next_index >= self.items.len() {
            self.recall_index = None;
            return None;
        }
        self.recall_index = Some(next_index);
        self.items.get(next_index)
    }
}

impl<T: Clone> HistoryBuffer<T> {
    pub fn page_after(&self, after: Option<u64>, limit: usize) -> HistoryPage<T> {
        let mut items = Vec::new();
        let mut remaining = limit.max(1);

        for record in &self.items {
            if after.is_some_and(|seq| record.seq <= seq) {
                continue;
            }
            items.push(record.clone());
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }

        let next_after = items.last().map(|record| record.seq);
        HistoryPage {
            items,
            next_after,
            oldest_available: self.oldest_seq(),
            newest_available: self.newest_seq(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HistoryBuffer;

    #[test]
    fn history_buffer_is_bounded() {
        let mut history = HistoryBuffer::try_with_capacity(2).expect("non-zero history");
        history.push("one");
        history.push("two");
        history.push("three");

        let page = history.page_after(None, 10);
        let values = page
            .items
            .into_iter()
            .map(|record| record.item)
            .collect::<Vec<_>>();

        assert_eq!(values, vec!["two", "three"]);
        assert_eq!(page.oldest_available, Some(2));
        assert_eq!(page.newest_available, Some(3));
    }

    #[test]
    fn history_buffer_recalls_previous_and_next_records() {
        let mut history = HistoryBuffer::try_with_capacity(4).expect("non-zero history");
        history.push("one");
        history.push("two");
        history.push("three");

        assert_eq!(history.prev().map(|record| record.item), Some("three"));
        assert_eq!(history.prev().map(|record| record.item), Some("two"));
        assert_eq!(history.prev().map(|record| record.item), Some("one"));
        assert_eq!(history.prev().map(|record| record.item), Some("one"));
        assert_eq!(history.next().map(|record| record.item), Some("two"));
        assert_eq!(history.next().map(|record| record.item), Some("three"));
        assert!(history.next().is_none());
    }

    #[test]
    fn history_buffer_push_and_clear_reset_recall() {
        let mut history = HistoryBuffer::try_with_capacity(2).expect("non-zero history");
        history.push("one");
        history.push("two");
        assert_eq!(history.prev().map(|record| record.item), Some("two"));
        history.push("three");
        assert_eq!(history.prev().map(|record| record.item), Some("three"));
        history.clear();
        assert!(history.prev().is_none());
    }

    #[test]
    fn history_buffer_pages_after_cursor() {
        let mut history = HistoryBuffer::try_with_capacity(4).expect("non-zero history");
        let one = history.push("one");
        let two = history.push("two");
        let _three = history.push("three");

        let page = history.page_after(Some(one), 10);
        let values = page
            .items
            .into_iter()
            .map(|record| record.item)
            .collect::<Vec<_>>();

        assert_eq!(values, vec!["two", "three"]);
        assert_eq!(page.next_after, Some(two + 1));
    }

    #[test]
    fn history_buffer_rejects_zero_capacity() {
        let err = HistoryBuffer::<&str>::try_with_capacity(0).expect_err("zero capacity rejected");
        assert_eq!(err.to_string(), "history capacity must be > 0");
    }
}
