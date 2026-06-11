use axum::http::HeaderMap;

use super::ApiError;

const TOKEN_ENV: &str = "MOTLIE_GATEWAY_API_TOKEN";

pub fn require_control_api_auth(headers: &HeaderMap) -> Result<(), ApiError> {
    let Ok(expected) = std::env::var(TOKEN_ENV) else {
        return Ok(());
    };
    if expected.trim().is_empty() {
        return Ok(());
    }

    let Some(value) = headers.get(axum::http::header::AUTHORIZATION) else {
        return Err(ApiError::unauthorized("missing bearer token"));
    };
    let Ok(value) = value.to_str() else {
        return Err(ApiError::unauthorized("invalid authorization header"));
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return Err(ApiError::unauthorized("expected bearer token"));
    };
    if token == expected {
        Ok(())
    } else {
        Err(ApiError::unauthorized("invalid bearer token"))
    }
}
