//! Graph business logic (single source of truth for ops).
//!
//! MutationExecutor and QueryExecutor impls should delegate to these helpers.

pub(crate) mod edge;
pub(crate) mod fragment;
pub(crate) mod name;
pub(crate) mod node;
pub(crate) mod summary;

#[cfg(test)]
mod tests;
