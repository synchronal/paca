pub mod endpoint;
pub mod manifest;

use std::env;

use reqwest::blocking::Client;
use reqwest::header::HeaderMap;
use reqwest::redirect;

use crate::error::PacaError;

/// User agent string used for HTTP requests
const USER_AGENT: &str = "llama-cpp";

/// Information resolved from a HEAD request to a HuggingFace file URL
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolveInfo {
    /// SHA-256 hash of the blob content (from X-Linked-Etag or ETag header)
    pub blob_hash: String,
    /// Commit hash for this revision (from X-Repo-Commit header)
    pub commit_hash: String,
}

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

/// Builds a Client configured for resolve-info HEAD requests (no redirect following).
pub fn resolve_client() -> Result<Client, PacaError> {
    Ok(Client::builder()
        .default_headers(default_headers())
        .redirect(redirect::Policy::none())
        .build()?)
}

pub fn fetch_resolve_info(client: &Client, url: &str) -> Result<ResolveInfo, PacaError> {
    let response = client.head(url).send()?;
    let headers = response.headers();

    let commit_hash = headers
        .get("x-repo-commit")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .ok_or_else(|| PacaError::MissingCommitHash(url.to_string()))?;

    let blob_hash = headers
        .get("x-linked-etag")
        .or_else(|| headers.get("etag"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| PacaError::MissingBlobHash(url.to_string()))?;

    Ok(ResolveInfo {
        blob_hash,
        commit_hash,
    })
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
