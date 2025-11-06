# Reader requirements

Design pattern follows libs/db/mutation.rs and libs/db/writer.rs in terms of
- Use enum with same visiblity to model Query.
- Use struct to model query args
- reader.rs contains client-side abstractions
- query.rs contains the server-side abstractions, query types, processor traits.
- graph.rs contains the actual graph-implementation for the queries.

## Query
- NodeById(NodeById)
- EdgeBySrcDstName(EdgeBySrcDstName)
- FragmentById(FragmentById)

- Each query type should have fields that correspond to the query input. For example, the query
to find node by ID (NodeById) should have a field of type Id (not Option<Id>). The client is
responsible for providing all required fields when constructing the query. The type system
ensures all required fields are present - no runtime validation is needed.
- Each query type should have a timestamp.
- Each query communicates results back to the client via a oneshot channel. The client awaits
on the receiver side with a timeout. The processor sends the result via the sender side.
- Each query has a timeout that the client specifies so that if exceeded, the client's await
returns with a timeout error.  

## Processing of queries
- Queries are sent by the clients over queue that supports multiple senders and multiple consumers (MPMC).
  - Senders send the Query and 1 of N receiver is selected to process  the Query.
  -  Exactly 1 receiver is allowed to process a Query.
- Similar to mutation::Processor,
  - There must be an interfaced defined for query processing -- query::Processor
  - The queue consumer is initialized with a readonly version of Storage.

# Implementation
- Use the 'flume' crate to implement MPMC semantics, along with standand async-std, tokio abstractions.
- Focus on idiomatic Rust and consistency with the design for mutations.

## Testing
- Follow the same pattern as mutations.  Place the tests in the same .rs files as the testing for
mutations.

