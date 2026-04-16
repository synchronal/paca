use thiserror::Error;

/// Errors that can occur during model download operations
#[derive(Debug, Error)]
pub enum PacaError {
    /// Failed to fetch the model manifest from HuggingFace
    #[error("Failed to fetch manifest: {0}")]
    ManifestFetch(#[from] reqwest::Error),

    /// Failed to parse the manifest JSON response
    #[error("Failed to parse manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    /// No downloadable files found in the model manifest
    #[error("No downloadable files found in manifest")]
    NoFiles,

    /// Failed to create or access the cache directory
    #[error("Failed to create cache directory: {0}")]
    CacheDir(std::io::Error),

    /// Failed to read response body during download
    #[error("Failed to download file: {0}")]
    Download(std::io::Error),

    /// Failed to delete a file from the cache
    #[error("Failed to delete file: {0}")]
    FileDelete(std::io::Error),

    /// Failed to write downloaded files to disk
    #[error("Failed to write file: {0}")]
    FileWrite(std::io::Error),

    /// Missing blob hash (ETag) from registry response
    #[error("Missing blob hash (ETag) for: {0}")]
    MissingBlobHash(String),

    /// Missing commit hash from registry response
    #[error("Missing x-repo-commit header for: {0}")]
    MissingCommitHash(String),

    /// Invalid model reference format
    #[error("{0}")]
    ModelRef(#[from] ModelRefError),

    /// Rate limited by the server (429), with optional Retry-After seconds
    #[error("Rate limited (retry after {0}s)")]
    RateLimited(u64),

    /// Invalid path format
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Failed to create a symlink
    #[error("Failed to create symlink: {0}")]
    Symlink(std::io::Error),
}

impl From<std::io::Error> for PacaError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => {
                PacaError::InvalidPath(format!("File not found: {}", err))
            }
            _ => PacaError::CacheDir(err),
        }
    }
}

/// Errors that can occur while parsing model references
#[derive(Debug, Error)]
pub enum ModelRefError {
    /// Missing tag component (expected format: owner/model:tag)
    #[error("Invalid model reference: missing tag (expected format: owner/model:tag)")]
    MissingTag,

    /// Missing owner component (expected format: owner/model:tag)
    #[error("Invalid model reference: missing owner (expected format: owner/model:tag)")]
    MissingOwner,
}
