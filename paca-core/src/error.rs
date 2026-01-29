use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("Failed to fetch manifest: {0}")]
    ManifestFetch(#[from] reqwest::Error),

    #[error("Failed to parse manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    #[error("No GGUF file found in manifest")]
    NoGgufFile,

    #[error("Failed to create cache directory: {0}")]
    CacheDir(std::io::Error),

    #[error("Failed to write file: {0}")]
    FileWrite(std::io::Error),

    #[error("{0}")]
    ModelRef(#[from] ModelRefError),
}

#[derive(Debug, Error)]
pub enum ModelRefError {
    #[error("Invalid model reference: missing tag (expected format: owner/model:tag)")]
    MissingTag,

    #[error("Invalid model reference: missing owner (expected format: owner/model:tag)")]
    MissingOwner,
}
