mod endpoint;
mod manifest;
mod model_ref;

pub use crate::error::DownloadError;

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;

use endpoint::get_model_endpoint;
use manifest::{fetch_manifest, manifest_filename};
use model_ref::ModelRef;

pub fn download_model(model: &str) -> Result<PathBuf, DownloadError> {
    let model_ref = ModelRef::parse(model)?;
    let manifest = fetch_manifest(&model_ref)?;

    let cache_dir = get_cache_dir()?;
    let filename = cache_filename(&model_ref, &manifest.gguf_file);
    let file_path = cache_dir.join(&filename);

    let endpoint = get_model_endpoint();
    let url = format!(
        "{}/{}/resolve/main/{}",
        endpoint,
        model_ref.repo(),
        manifest.gguf_file
    );

    let remote_etag = fetch_etag(&url)?;

    if file_path.exists() && etag_matches(&cache_dir, &filename, &remote_etag) {
        return Ok(file_path);
    }

    save_etag(&cache_dir, &filename, &remote_etag)?;
    download_file(&url, &file_path)?;
    save_manifest(&cache_dir, &model_ref, &manifest.raw_json)?;

    Ok(file_path)
}

fn cache_filename(model_ref: &ModelRef, gguf_file: &str) -> String {
    let flat_gguf = gguf_file.replace('/', "_");
    format!("{}_{}_{}", model_ref.owner, model_ref.model, flat_gguf)
}

fn save_manifest(
    cache_dir: &std::path::Path,
    model_ref: &ModelRef,
    raw_json: &str,
) -> Result<(), DownloadError> {
    let manifest_path = cache_dir.join(manifest_filename(model_ref));
    fs::write(&manifest_path, raw_json).map_err(DownloadError::FileWrite)?;
    Ok(())
}

fn fetch_etag(url: &str) -> Result<String, DownloadError> {
    let client = Client::builder().redirect(Policy::none()).build()?;

    let response = client.head(url).header("User-Agent", "llama-cpp").send()?;

    let etag = response
        .headers()
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    Ok(etag)
}

fn etag_matches(cache_dir: &std::path::Path, filename: &str, remote_etag: &str) -> bool {
    let etag_path = cache_dir.join(format!("{}.etag", filename));
    fs::read_to_string(etag_path)
        .map(|local_etag| local_etag == remote_etag)
        .unwrap_or(false)
}

fn save_etag(cache_dir: &std::path::Path, filename: &str, etag: &str) -> Result<(), DownloadError> {
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

fn download_file(url: &str, path: &PathBuf) -> Result<(), DownloadError> {
    let client = Client::new();
    let mut response = client
        .get(url)
        .header("User-Agent", "llama-cpp")
        .send()?
        .error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);

    let progress_bar = ProgressBar::new(total_size);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut file = File::create(path).map_err(DownloadError::FileWrite)?;
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
    fn get_cache_dir_returns_llama_cpp_subdirectory() {
        let cache_dir = get_cache_dir().unwrap();
        assert!(cache_dir.ends_with("llama.cpp"));
    }

    #[test]
    fn download_model_returns_error_for_missing_tag() {
        let result = download_model("owner/model");
        assert!(result.is_err());
    }

    #[test]
    fn cache_filename_flattens_simple_gguf_file() {
        let model_ref = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL").unwrap();
        let filename = cache_filename(&model_ref, "GLM-4.7-Flash-UD-Q2_K_XL.gguf");
        assert_eq!(
            filename,
            "unsloth_GLM-4.7-Flash-GGUF_GLM-4.7-Flash-UD-Q2_K_XL.gguf"
        );
    }

    #[test]
    fn cache_filename_flattens_subdirectory_gguf_file() {
        let model_ref = ModelRef::parse("unsloth/GLM-4.7-Flash-GGUF:BF16").unwrap();
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
}
