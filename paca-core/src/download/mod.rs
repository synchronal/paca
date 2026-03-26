pub use crate::error::PacaError;

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;

use crate::cache::{cache_filename, etag_matches, get_cache_dir, save_etag, save_manifest};
use crate::model::ModelRef;
use crate::registry::default_headers;
use crate::registry::endpoint::get_model_endpoint;
use crate::registry::fetch_etag;
use crate::registry::manifest::fetch_manifest;

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
        let bytes_read = response.read(&mut buffer).map_err(PacaError::Download)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])
            .map_err(PacaError::FileWrite)?;
        progress_bar.inc(bytes_read as u64);
    }

    file.flush().map_err(PacaError::FileWrite)?;
    progress_bar.finish_with_message("Download complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_model_returns_error_for_missing_tag() {
        let result = download_model("owner/model", None);
        assert!(result.is_err());
    }
}
