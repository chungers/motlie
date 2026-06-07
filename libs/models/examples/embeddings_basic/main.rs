use anyhow::Result;

#[path = "../embedding_example_support.rs"]
mod embedding_example_support;
#[path = "../support.rs"]
mod support;

#[tokio::main]
async fn main() -> Result<()> {
    embedding_example_support::run("embeddings_basic").await
}
