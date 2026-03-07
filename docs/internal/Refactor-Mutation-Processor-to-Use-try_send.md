# Refactor Mutation Processor to Use try_send() for Forwarding

## Problem
The mutation processor should use `try_send()` when forwarding messages to the next entity in the chain. This will ensure non-blocking behavior and improved performance. Currently, the implementation in `mutation.rs` line 180 does not use `try_send()`.

## Proposed Solution
Replace the existing forwarding mechanism with `try_send()` in the affected code. Ensure that appropriate error handling mechanisms are in place to deal with potential message failures.

## File Reference
- File: `libs/db/src/mutation.rs`
- Line: [180](https://github.com/chungers/motlie/blob/main/libs/db/src/mutation.rs#L180)

## Steps to Implement
1. Refactor the forwarding logic in the mutation processor.
2. Replace the existing mechanism with `try_send()`.
3. Add tests to ensure that the functionality is working as expected and handles errors.

## Expected Outcome
The mutation processor should forward messages using `try_send()`, ensuring better handling of scenarios where immediate message sending might block or fail.