use std::collections::BTreeMap;

use crate::{Result, VoiceError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequencedFrame<T> {
    pub sequence: u64,
    pub payload: T,
}

#[derive(Clone, Debug)]
pub struct SequencedFrameReorder<T> {
    next_expected: u64,
    capacity: usize,
    pending: BTreeMap<u64, T>,
}

impl<T> SequencedFrameReorder<T> {
    pub fn new(first_sequence: u64, capacity: usize) -> Self {
        Self {
            next_expected: first_sequence,
            capacity,
            pending: BTreeMap::new(),
        }
    }

    pub fn push(&mut self, frame: SequencedFrame<T>) -> Result<Vec<SequencedFrame<T>>> {
        if frame.sequence < self.next_expected {
            return Err(VoiceError::StaleFrameSequence {
                sequence: frame.sequence,
                next_expected: self.next_expected,
            });
        }

        let distance = frame.sequence.saturating_sub(self.next_expected) as usize;
        if distance > self.capacity {
            return Err(VoiceError::ReorderCapacityExceeded {
                sequence: frame.sequence,
                next_expected: self.next_expected,
                capacity: self.capacity,
            });
        }

        self.pending.entry(frame.sequence).or_insert(frame.payload);
        Ok(self.drain_ready())
    }

    pub fn flush(self) -> Vec<SequencedFrame<T>> {
        self.pending
            .into_iter()
            .map(|(sequence, payload)| SequencedFrame { sequence, payload })
            .collect()
    }

    fn drain_ready(&mut self) -> Vec<SequencedFrame<T>> {
        let mut ready = Vec::new();
        while let Some(payload) = self.pending.remove(&self.next_expected) {
            ready.push(SequencedFrame {
                sequence: self.next_expected,
                payload,
            });
            self.next_expected += 1;
        }
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reorders_until_contiguous_sequence_is_ready() {
        let mut reorder = SequencedFrameReorder::new(1, 4);
        assert_eq!(
            reorder
                .push(SequencedFrame {
                    sequence: 2,
                    payload: "b"
                })
                .expect("frame should buffer"),
            Vec::<SequencedFrame<&str>>::new()
        );
        let ready = reorder
            .push(SequencedFrame {
                sequence: 1,
                payload: "a",
            })
            .expect("frame should release contiguous frames");

        assert_eq!(
            ready,
            vec![
                SequencedFrame {
                    sequence: 1,
                    payload: "a"
                },
                SequencedFrame {
                    sequence: 2,
                    payload: "b"
                }
            ]
        );
    }

    #[test]
    fn rejects_stale_frames() {
        let mut reorder = SequencedFrameReorder::new(1, 4);
        let _ = reorder
            .push(SequencedFrame {
                sequence: 1,
                payload: "a",
            })
            .expect("first frame should pass");

        let err = reorder
            .push(SequencedFrame {
                sequence: 1,
                payload: "duplicate",
            })
            .expect_err("duplicate should be stale");

        assert!(matches!(
            err,
            VoiceError::StaleFrameSequence {
                sequence: 1,
                next_expected: 2
            }
        ));
    }
}
