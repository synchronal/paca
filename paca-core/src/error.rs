use thiserror::Error;

/// Errors that can occur during model download operations
#[derive(Debug, Error)]
pub enum DownloadError {
    /// Failed to fetch the model manifest from HuggingFace
    #[error("Failed to fetch manifest: {0}")]
    ManifestFetch(#[from] reqwest::Error),

    /// Failed to parse the manifest JSON response
    #[error("Failed to parse manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    /// No GGUF file found in the model manifest
    #[error("No GGUF file found in manifest")]
    NoGgufFile,

    /// Failed to create or access the cache directory
    #[error("Failed to create cache directory: {0}")]
    CacheDir(std::io::Error),

    /// Failed to write downloaded files to disk
    #[error("Failed to write file: {0}")]
    FileWrite(std::io::Error),

    /// Invalid model reference format
    #[error("{0}")]
    ModelRef(#[from] ModelRefError),
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
