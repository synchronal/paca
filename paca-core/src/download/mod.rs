mod endpoint;
mod manifest;
mod model_ref;

pub use crate::error::DownloadError;

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::HeaderMap;
use reqwest::redirect::Policy;

use endpoint::get_model_endpoint;
use manifest::{fetch_manifest, manifest_filename};
use model_ref::ModelRef;

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
pub fn download_model(
    model: &str,
    cache_dir: Option<PathBuf>,
) -> Result<Vec<PathBuf>, DownloadError> {
    let model_ref: ModelRef = model.parse()?;
    let headers = default_headers();
    let client = Client::builder().default_headers(headers.clone()).build()?;
    let etag_client = Client::builder()
        .default_headers(headers)
        .redirect(Policy::none())
        .build()?;

    let manifest = fetch_manifest(&client, &model_ref)?;
    let cache_dir = match cache_dir {
        Some(dir) => {
            fs::create_dir_all(&dir).map_err(DownloadError::CacheDir)?;
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

        let remote_etag = fetch_etag(&etag_client, &url)?;

        if file_path.exists() && etag_matches(&cache_dir, &filename, &remote_etag) {
            let existing_size = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

            if existing_size >= gguf_file.size {
                paths.push(file_path);
                continue;
            }

            download_file(&client, &url, &file_path, existing_size)?;
        } else {
            save_etag(&cache_dir, &filename, &remote_etag)?;
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

fn save_manifest(
    cache_dir: &Path,
    model_ref: &ModelRef,
    raw_json: &str,
) -> Result<(), DownloadError> {
    let manifest_path = cache_dir.join(manifest_filename(model_ref));
    fs::write(&manifest_path, raw_json).map_err(DownloadError::FileWrite)?;
    Ok(())
}

fn fetch_etag(client: &Client, url: &str) -> Result<String, DownloadError> {
    let response = client.head(url).send()?;

    let etag = response
        .headers()
        .get("x-linked-etag")
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

fn save_etag(cache_dir: &Path, filename: &str, etag: &str) -> Result<(), DownloadError> {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::write(&etag_path, etag).map_err(DownloadError::FileWrite)?;
    Ok(())
}

fn get_cache_dir() -> Result<PathBuf, DownloadError> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| {
            DownloadError::CacheDir(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine cache directory",
            ))
        })?
        .join("llama.cpp");

    fs::create_dir_all(&cache_dir).map_err(DownloadError::CacheDir)?;

    Ok(cache_dir)
}

fn download_file(
    client: &Client,
    url: &str,
    path: &Path,
    resume_from: u64,
) -> Result<(), DownloadError> {
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
            .map_err(DownloadError::FileWrite)?;
        (file, resume_from)
    } else {
        (File::create(path).map_err(DownloadError::FileWrite)?, 0)
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

    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = response
            .read(&mut buffer)
            .map_err(DownloadError::FileWrite)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .map_err(DownloadError::FileWrite)?;
        progress_bar.inc(bytes_read as u64);
    }

    progress_bar.finish_with_message("Download complete");

    Ok(())
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
