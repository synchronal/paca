mod endpoint;
mod manifest;
mod model_ref;

pub use crate::error::PacaError;

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::HeaderMap;

use endpoint::get_model_endpoint;
use manifest::{fetch_manifest, manifest_filename};
use model_ref::ModelRef;

/// Information about a downloaded model
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ModelInfo {
    /// The model reference (owner/model:tag)
    pub model_ref: ModelRef,
    /// Whether the model is installed
    pub installed: bool,
}

/// User agent string used for HTTP requests
const USER_AGENT: &str = "llama-cpp";

fn default_headers() -> HeaderMap {
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

/// Downloads a GGUF model from HuggingFace with support for resumable downloads
/// and incremental updates using ETag validation
pub fn download_model(model: &str, cache_dir: Option<PathBuf>) -> Result<Vec<PathBuf>, PacaError> {
    let model_ref: ModelRef = model.parse()?;
    let headers = default_headers();
    let client = Client::builder().default_headers(headers).build()?;

    let manifest = fetch_manifest(&client, &model_ref)?;
    let cache_dir = match cache_dir {
        Some(dir) => {
            fs::create_dir_all(&dir).map_err(PacaError::CacheDir)?;
            dir
        }
        None => get_cache_dir()?,
    };
    let endpoint = get_model_endpoint();

    let mut paths = Vec::new();

    for gguf_file in &manifest.gguf_files {
        let filename = cache_filename(&model_ref, &gguf_file.filename);
        let file_path = cache_dir.join(&filename);

        let url = format!(
            "{}/{}/resolve/main/{}",
            endpoint,
            model_ref.repo(),
            gguf_file.filename
        );

        let remote_etag = fetch_etag(&client, &url)?;

        if !etag_matches(&cache_dir, &filename, &remote_etag) {
            save_etag(&cache_dir, &filename, &remote_etag)?;
        }

        if file_path.exists() {
            let existing_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            if existing_size >= gguf_file.size {
                paths.push(file_path);
                continue;
            }

            download_file(&client, &url, &file_path, existing_size)?;
        } else {
            download_file(&client, &url, &file_path, 0)?;
        }

        paths.push(file_path);
    }

    save_manifest(&cache_dir, &model_ref, &manifest.raw_json)?;

    Ok(paths)
}

fn cache_filename(model_ref: &ModelRef, gguf_file: &str) -> String {
    let flat_gguf = gguf_file.replace('/', "_");
    format!("{}_{}_{}", model_ref.owner, model_ref.model, flat_gguf)
}

fn save_manifest(cache_dir: &Path, model_ref: &ModelRef, raw_json: &str) -> Result<(), PacaError> {
    let manifest_path = cache_dir.join(manifest_filename(model_ref));
    fs::write(&manifest_path, raw_json).map_err(PacaError::FileWrite)?;
    Ok(())
}

fn fetch_etag(client: &Client, url: &str) -> Result<String, PacaError> {
    let response = client.head(url).send()?;

    let etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    Ok(etag)
}

fn etag_matches(cache_dir: &Path, filename: &str, remote_etag: &str) -> bool {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::read_to_string(etag_path)
        .map(|local_etag| local_etag == remote_etag)
        .unwrap_or(false)
}

fn save_etag(cache_dir: &Path, filename: &str, etag: &str) -> Result<(), PacaError> {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::write(&etag_path, etag).map_err(PacaError::FileWrite)?;
    Ok(())
}

fn get_cache_dir() -> Result<PathBuf, PacaError> {
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

fn download_file(
    client: &Client,
    url: &str,
    path: &Path,
    resume_from: u64,
) -> Result<(), PacaError> {
    let mut request = client.get(url);

    if resume_from > 0 {
        request = request.header("Range", format!("bytes={}-", resume_from));
    }

    let mut response = request.send()?.error_for_status()?;

    let is_partial = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;

    let (mut file, start_pos) = if is_partial {
        let file = fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(PacaError::FileWrite)?;
        (BufWriter::new(file), resume_from)
    } else {
        (
            BufWriter::new(File::create(path).map_err(PacaError::FileWrite)?),
            0,
        )
    };

    let total_size = response.content_length().unwrap_or(0) + start_pos;

    let progress_bar = ProgressBar::new(total_size);
    progress_bar.set_position(start_pos);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut buffer = [0u8; 131072];

    loop {
        let bytes_read = response.read(&mut buffer).map_err(PacaError::FileWrite)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .map_err(PacaError::FileWrite)?;
        progress_bar.inc(bytes_read as u64);
    }

    progress_bar.finish_with_message("Download complete");

    Ok(())
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

/// Parses a model manifest file and returns model information
fn parse_model_manifest(path: &Path) -> Result<ModelInfo, PacaError> {
    let content = fs::read_to_string(path)?;
    let _json: serde_json::Value = serde_json::from_str(&content)?;

    let model_ref = extract_model_ref(path)?;

    Ok(ModelInfo {
        model_ref,
        installed: true,
    })
}

/// Extracts the model reference from a manifest filename
fn extract_model_ref(manifest_path: &Path) -> Result<ModelRef, PacaError> {
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

    #[test]
    fn get_cache_dir_returns_llama_cpp_subdirectory() {
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.ends_with("llama.cpp"));
    }

    #[test]
    fn download_model_returns_error_for_missing_tag() {
        let result = download_model("owner/model", None);
        assert!(result.is_err());
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
