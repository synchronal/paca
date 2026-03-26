pub mod endpoint;
pub mod manifest;

use std::env;

use reqwest::blocking::Client;
use reqwest::header::HeaderMap;

use crate::error::PacaError;

/// User agent string used for HTTP requests
const USER_AGENT: &str = "llama-cpp";

pub fn default_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("User-Agent", USER_AGENT.parse().unwrap());

    if let Ok(token) = env::var("HF_TOKEN") {
        headers.insert(
            "Authorization",
            format!("Bearer {}", token).parse().unwrap(),
        );
    }

    headers
}

pub fn fetch_etag(client: &Client, url: &str) -> Result<String, PacaError> {
    let response = client.head(url).send()?;

    let etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    Ok(etag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_headers_includes_authorization_when_hf_token_set() {
        temp_env::with_vars([("HF_TOKEN", Some("test-token-123"))], || {
            let headers = default_headers();
            assert_eq!(
                headers.get("Authorization").unwrap().to_str().unwrap(),
                "Bearer test-token-123"
            );
        });
    }

    #[test]
    fn default_headers_excludes_authorization_when_hf_token_unset() {
        temp_env::with_vars_unset(["HF_TOKEN"], || {
            let headers = default_headers();
            assert!(headers.get("Authorization").is_none());
        });
    }

    #[test]
    fn default_headers_includes_user_agent() {
        temp_env::with_vars_unset(["HF_TOKEN"], || {
            let headers = default_headers();
            assert_eq!(
                headers.get("User-Agent").unwrap().to_str().unwrap(),
                "llama-cpp"
            );
        });
    }
}
