use std::future::Future;
use std::pin::Pin;

use anyhow::{Context, Result};
use motlie_model::BundleHandle;

pub async fn run_with_shutdown<H, F, T>(handle: H, body: F) -> Result<T>
where
    H: BundleHandle,
    F: for<'a> FnOnce(&'a H) -> Pin<Box<dyn Future<Output = Result<T>> + 'a>>,
{
    let body_result = body(&handle).await;
    let shutdown_result = handle.shutdown().await.context("shutdown failed");

    match (body_result, shutdown_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
        (Err(body_error), Err(shutdown_error)) => {
            Err(body_error.context(format!("additionally, {shutdown_error:#}")))
        }
    }
}
