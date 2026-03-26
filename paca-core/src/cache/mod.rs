pub use crate::error::PacaError;

use std::fs;
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;

use crate::model::ModelRef;
use crate::registry::default_headers;
use crate::registry::endpoint::get_model_endpoint;
use crate::registry::fetch_etag;
use crate::registry::manifest::{fetch_manifest, manifest_filename};

/// Information about a downloaded model
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ModelInfo {
    /// The model reference (owner/model:tag)
    pub model_ref: ModelRef,
    /// Whether the model is installed
    pub installed: bool,
}

/// Information about a model file with an outdated etag
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OutdatedModelInfo {
    /// The model reference (owner/model:tag)
    pub model_ref: ModelRef,
    /// The filename of the outdated file
    pub filename: String,
    /// The local file path
    pub file_path: PathBuf,
}

/// Lists all downloaded models from the cache directory
pub fn list_models(cache_dir: Option<PathBuf>) -> Result<Vec<ModelInfo>, PacaError> {
    let cache_dir = match cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && path.extension().is_some_and(|ext| ext == "json")
            && let Ok(manifest) = parse_model_manifest(&path)
        {
            models.push(manifest);
        }
    }

    models.sort_by(|a, b| a.model_ref.to_string().cmp(&b.model_ref.to_string()));

    Ok(models)
}

/// Checks which downloaded models have outdated files
pub fn check_outdated_models(
    cache_dir: Option<PathBuf>,
) -> Result<Vec<OutdatedModelInfo>, PacaError> {
    let cache_dir = match cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let client = Client::builder()
        .default_headers(default_headers())
        .build()?;
    let endpoint = get_model_endpoint();
    let mut outdated_models = Vec::new();

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && path.extension().is_some_and(|ext| ext == "json")
            && let Ok(model_ref) = extract_model_ref(&path)
        {
            let manifest = fetch_manifest(&client, &model_ref)?;

            for gguf_file in &manifest.gguf_files {
                let filename = cache_filename(&model_ref, &gguf_file.filename);
                let file_path = cache_dir.join(&filename);
                let url = format!(
                    "{}/{}/resolve/main/{}",
                    endpoint,
                    model_ref.repo(),
                    gguf_file.filename
                );

                let remote_etag = fetch_etag(&client, &url);

                if let Ok(remote_etag) = remote_etag
                    && !etag_matches(&cache_dir, &filename, &remote_etag)
                {
                    outdated_models.push(OutdatedModelInfo {
                        model_ref: model_ref.clone(),
                        filename: gguf_file.filename.clone(),
                        file_path,
                    });
                }
            }
        }
    }

    outdated_models.sort_by(|a, b| a.model_ref.to_string().cmp(&b.model_ref.to_string()));

    Ok(outdated_models)
}

pub(crate) fn get_cache_dir() -> Result<PathBuf, PacaError> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| {
            PacaError::CacheDir(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine cache directory",
            ))
        })?
        .join("llama.cpp");

    fs::create_dir_all(&cache_dir).map_err(PacaError::CacheDir)?;

    Ok(cache_dir)
}

pub(crate) fn cache_filename(model_ref: &ModelRef, gguf_file: &str) -> String {
    let flat_gguf = gguf_file.replace('/', "_");
    format!("{}_{}_{}", model_ref.owner, model_ref.model, flat_gguf)
}

pub(crate) fn save_manifest(
    cache_dir: &Path,
    model_ref: &ModelRef,
    raw_json: &str,
) -> Result<(), PacaError> {
    let manifest_path = cache_dir.join(manifest_filename(model_ref));
    fs::write(&manifest_path, raw_json).map_err(PacaError::FileWrite)?;
    Ok(())
}

pub(crate) fn etag_matches(cache_dir: &Path, filename: &str, remote_etag: &str) -> bool {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::read_to_string(etag_path)
        .map(|local_etag| local_etag == remote_etag)
        .unwrap_or(false)
}

pub(crate) fn save_etag(cache_dir: &Path, filename: &str, etag: &str) -> Result<(), PacaError> {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::write(&etag_path, etag).map_err(PacaError::FileWrite)?;
    Ok(())
}

fn parse_model_manifest(path: &Path) -> Result<ModelInfo, PacaError> {
    let content = fs::read_to_string(path)?;
    let _json: serde_json::Value = serde_json::from_str(&content)?;

    let model_ref = extract_model_ref(path)?;

    Ok(ModelInfo {
        model_ref,
        installed: true,
    })
}

pub(crate) fn extract_model_ref(manifest_path: &Path) -> Result<ModelRef, PacaError> {
    let filename = manifest_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| PacaError::InvalidPath("Invalid filename".to_string()))?;

    let parts: Vec<&str> = filename
        .strip_prefix("manifest=")
        .unwrap_or(filename)
        .split('=')
        .collect();

    if parts.len() != 3 {
        return Err(PacaError::InvalidPath(
            "Invalid manifest filename format".to_string(),
        ));
    }

    let owner = parts[0].to_string();
    let model = parts[1].to_string();
    let tag = parts[2]
        .strip_suffix(".json")
        .unwrap_or(parts[2])
        .to_string();

    Ok(ModelRef { owner, model, tag })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_cache_dir_returns_llama_cpp_subdirectory() {
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.ends_with("llama.cpp"));
    }

    #[test]
    fn cache_filename_flattens_simple_gguf_file() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        let filename = cache_filename(&model_ref, "GLM-4.7-Flash-UD-Q2_K_XL.gguf");
        assert_eq!(
            filename,
            "unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-UD-Q2_K_XL.gguf"
        );
    }

    #[test]
    fn cache_filename_flattens_subdirectory_gguf_file() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:BF16".parse().unwrap();
        let filename = cache_filename(&model_ref, "BF16/GLM-4.7-Flash-BF16-00001-of-00002.gguf");
        assert_eq!(
            filename,
            "unsloth_GLM-4.7-Flash-GGUF_BF16_GLM-4.7-Flash-BF16-00001-of-00002.gguf"
        );
    }

    #[test]
    fn etag_matches_returns_true_when_etag_matches() {
        let dir = tempfile::tempdir().unwrap();
        let filename = "model.gguf";
        let etag = "\"abc123\"";

        save_etag(dir.path(), filename, etag).unwrap();
        assert!(etag_matches(dir.path(), filename, etag));
    }

    #[test]
    fn etag_matches_returns_false_when_etag_differs() {
        let dir = tempfile::tempdir().unwrap();
        let filename = "model.gguf";

        save_etag(dir.path(), filename, "\"old\"").unwrap();
        assert!(!etag_matches(dir.path(), filename, "\"new\""));
    }

    #[test]
    fn etag_matches_returns_false_when_no_etag_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!etag_matches(dir.path(), "model.gguf", "\"abc123\""));
    }

    #[test]
    fn etag_matches_returns_false_when_etag_file_corrupted() {
        let dir = tempfile::tempdir().unwrap();
        let filename = "model.gguf";
        let etag_path = dir.path().join(format!("{}.etag", filename));

        // Write invalid JSON to the etag file
        fs::write(&etag_path, "[invalid json").unwrap();

        assert!(!etag_matches(dir.path(), filename, "\"abc123\""));
    }
}
