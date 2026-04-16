pub use crate::error::PacaError;

use std::fs::{self, File};
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;

use crate::cache::{blob_exists, blobs_dir, get_hub_dir, save_ref, snapshots_dir};
use crate::model::ModelRef;
use crate::registry::default_headers;
use crate::registry::endpoint::get_model_endpoint;
use crate::registry::manifest::fetch_manifest;
use crate::registry::{fetch_resolve_info, resolve_client};
use crate::sysinfo::check_disk_space;

/// Downloads a GGUF model from HuggingFace into the HF Hub cache format
pub async fn download_model(
    model: &str,
    hub_dir: Option<PathBuf>,
) -> Result<Vec<PathBuf>, PacaError> {
    let model_ref: ModelRef = model.parse()?;
    let headers = default_headers();
    let client = Client::builder()
        .default_headers(headers)
        .tcp_keepalive(Duration::from_secs(15))
        .build()?;

    let manifest = fetch_manifest(&client, &model_ref).await?;
    let hub_dir = match hub_dir {
        Some(dir) => {
            fs::create_dir_all(&dir).map_err(PacaError::CacheDir)?;
            dir
        }
        None => get_hub_dir()?,
    };
    let endpoint = get_model_endpoint();
    let head_client = resolve_client()?;

    let blobs = blobs_dir(&hub_dir, &model_ref);
    fs::create_dir_all(&blobs).map_err(PacaError::CacheDir)?;

    let total_size: u64 = manifest.gguf_files.iter().map(|f| f.size).sum();
    check_disk_space(&blobs, total_size)?;

    let multi = MultiProgress::new();
    let bars: Vec<ProgressBar> = manifest
        .gguf_files
        .iter()
        .map(|gguf_file| {
            let bar = multi.add(ProgressBar::new(gguf_file.size));
            bar.set_style(pending_style());
            bar.set_message(gguf_file.filename.clone());
            bar
        })
        .collect();

    let mut set: tokio::task::JoinSet<Result<(PathBuf, String), PacaError>> =
        tokio::task::JoinSet::new();

    for (gguf_file, bar) in manifest.gguf_files.into_iter().zip(bars) {
        let client = client.clone();
        let head_client = head_client.clone();
        let endpoint = endpoint.clone();
        let model_ref = model_ref.clone();
        let hub_dir = hub_dir.clone();
        let blobs = blobs.clone();

        set.spawn(async move {
            let url = format!(
                "{}/{}/resolve/main/{}",
                endpoint,
                model_ref.repo(),
                gguf_file.filename
            );

            let resolve_info = fetch_resolve_info(&head_client, &url).await?;
            let blob_path = blobs.join(&resolve_info.blob_hash);

            if blob_exists(&hub_dir, &model_ref, &resolve_info.blob_hash) {
                let existing_size = fs::metadata(&blob_path).map(|m| m.len()).unwrap_or(0);

                if existing_size >= gguf_file.size {
                    bar.set_style(download_style());
                    bar.set_position(gguf_file.size);
                    bar.finish();

                    let symlink_path = create_snapshot_symlink(
                        &hub_dir,
                        &model_ref,
                        &resolve_info.commit_hash,
                        &gguf_file.filename,
                        &resolve_info.blob_hash,
                    )?;
                    return Ok((symlink_path, resolve_info.commit_hash));
                }

                // Partial blob exists — resume with single connection
                download_file(&client, &url, &blob_path, existing_size, &bar).await?;
            } else if gguf_file.size >= PARALLEL_THRESHOLD {
                download_file_parallel(&client, &url, &blob_path, gguf_file.size, &bar).await?;
            } else {
                download_file(&client, &url, &blob_path, 0, &bar).await?;
            }

            let symlink_path = create_snapshot_symlink(
                &hub_dir,
                &model_ref,
                &resolve_info.commit_hash,
                &gguf_file.filename,
                &resolve_info.blob_hash,
            )?;
            Ok((symlink_path, resolve_info.commit_hash))
        });
    }

    let mut paths = Vec::new();
    let mut commit_hash = None;

    while let Some(result) = set.join_next().await {
        let (path, hash) = result.expect("download task panicked")?;
        paths.push(path);
        if commit_hash.is_none() {
            commit_hash = Some(hash);
        }
    }

    if let Some(commit) = &commit_hash {
        save_ref(&hub_dir, &model_ref, commit)?;
    }

    Ok(paths)
}

fn create_snapshot_symlink(
    hub_dir: &Path,
    model_ref: &ModelRef,
    commit_hash: &str,
    filename: &str,
    blob_hash: &str,
) -> Result<PathBuf, PacaError> {
    let snapshot_dir = snapshots_dir(hub_dir, model_ref).join(commit_hash);

    let symlink_path = snapshot_dir.join(filename);
    if let Some(parent) = symlink_path.parent() {
        fs::create_dir_all(parent).map_err(PacaError::CacheDir)?;
    }

    // Calculate relative path from symlink to blob
    let depth = filename.matches('/').count() + 2;
    let relative_blob = format!("{}blobs/{}", "../".repeat(depth), blob_hash);

    // Remove existing symlink if present
    if symlink_path.exists() || symlink_path.symlink_metadata().is_ok() {
        fs::remove_file(&symlink_path).map_err(PacaError::FileDelete)?;
    }

    std::os::unix::fs::symlink(&relative_blob, &symlink_path).map_err(PacaError::Symlink)?;

    Ok(symlink_path)
}

const MAX_RETRIES: u32 = 5;

async fn download_file(
    client: &Client,
    url: &str,
    path: &Path,
    resume_from: u64,
    progress_bar: &ProgressBar,
) -> Result<(), PacaError> {
    let mut retries: u32 = 0;
    let mut bytes_on_disk = resume_from;

    loop {
        match attempt_download(client, url, path, bytes_on_disk, progress_bar).await {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable(&e) => {
                let new_size = fs::metadata(path).map(|m| m.len()).unwrap_or(bytes_on_disk);

                if new_size > bytes_on_disk {
                    retries = 0;
                    bytes_on_disk = new_size;
                } else {
                    retries += 1;
                }

                if retries > MAX_RETRIES {
                    return Err(e);
                }

                let delay_secs = match &e {
                    PacaError::RateLimited(wait) if *wait > 0 => *wait,
                    _ => 1u64 << retries,
                };
                progress_bar.println(format!(
                    "Download error: {e}. Retrying in {delay_secs}s (attempt {retries}/{MAX_RETRIES})..."
                ));
                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

fn is_retryable(error: &PacaError) -> bool {
    match error {
        PacaError::Download(_) => true,
        PacaError::ManifestFetch(e) => {
            if let Some(status) = e.status() {
                status.is_server_error()
            } else {
                true
            }
        }
        PacaError::RateLimited(_) => true,
        _ => false,
    }
}

async fn attempt_download(
    client: &Client,
    url: &str,
    path: &Path,
    resume_from: u64,
    progress_bar: &ProgressBar,
) -> Result<(), PacaError> {
    let mut request = client.get(url);

    if resume_from > 0 {
        request = request.header("Range", format!("bytes={}-", resume_from));
    }

    let response = request.send().await?;

    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        return Err(PacaError::RateLimited(retry_after));
    }

    let mut response = response.error_for_status()?;

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

    progress_bar.set_style(download_style());
    progress_bar.set_position(start_pos);

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PacaError::Download(std::io::Error::other(e)))?
    {
        file.write_all(&chunk).map_err(PacaError::FileWrite)?;
        progress_bar.inc(chunk.len() as u64);
    }

    file.flush().map_err(PacaError::FileWrite)?;
    progress_bar.finish();

    Ok(())
}

/// Minimum file size to use parallel chunk downloads (100 MB)
const PARALLEL_THRESHOLD: u64 = 100 * 1024 * 1024;

/// Number of concurrent connections per file
const CHUNK_COUNT: usize = 4;

async fn download_file_parallel(
    client: &Client,
    url: &str,
    path: &Path,
    total_size: u64,
    progress_bar: &ProgressBar,
) -> Result<(), PacaError> {
    let chunks = calculate_chunks(total_size, CHUNK_COUNT);

    let file = File::create(path).map_err(PacaError::FileWrite)?;
    file.set_len(total_size).map_err(PacaError::FileWrite)?;
    drop(file);

    progress_bar.set_style(download_style());
    progress_bar.set_position(0);

    let mut set = tokio::task::JoinSet::new();

    for (start, end) in chunks {
        let client = client.clone();
        let url = url.to_string();
        let path = path.to_path_buf();
        let bar = progress_bar.clone();

        set.spawn(async move { download_chunk(&client, &url, &path, start, end, &bar).await });
    }

    while let Some(result) = set.join_next().await {
        result.expect("chunk download task panicked")?;
    }

    progress_bar.finish();
    Ok(())
}

async fn download_chunk(
    client: &Client,
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
    progress_bar: &ProgressBar,
) -> Result<(), PacaError> {
    let mut retries: u32 = 0;
    let mut bytes_written: u64 = 0;
    let chunk_size = end - start + 1;

    loop {
        let current_start = start + bytes_written;
        match attempt_chunk_download(client, url, path, current_start, end, progress_bar).await {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable(&e) => {
                let new_bytes = chunk_bytes_on_disk(path, start, chunk_size);
                if new_bytes > bytes_written {
                    retries = 0;
                    bytes_written = new_bytes;
                } else {
                    retries += 1;
                }

                if retries > MAX_RETRIES {
                    return Err(e);
                }

                let delay_secs = match &e {
                    PacaError::RateLimited(wait) if *wait > 0 => *wait,
                    _ => 1u64 << retries,
                };
                progress_bar.println(format!(
                    "Chunk download error: {e}. Retrying in {delay_secs}s (attempt {retries}/{MAX_RETRIES})..."
                ));
                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

fn chunk_bytes_on_disk(path: &Path, start: u64, chunk_size: u64) -> u64 {
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if file_size <= start {
        0
    } else {
        (file_size - start).min(chunk_size)
    }
}

async fn attempt_chunk_download(
    client: &Client,
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
    progress_bar: &ProgressBar,
) -> Result<(), PacaError> {
    let response = client
        .get(url)
        .header("Range", format!("bytes={}-{}", start, end))
        .send()
        .await?;

    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        return Err(PacaError::RateLimited(retry_after));
    }

    let mut response = response.error_for_status()?;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(PacaError::FileWrite)?;
    file.seek(std::io::SeekFrom::Start(start))
        .map_err(PacaError::FileWrite)?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PacaError::Download(std::io::Error::other(e)))?
    {
        file.write_all(&chunk).map_err(PacaError::FileWrite)?;
        progress_bar.inc(chunk.len() as u64);
    }

    file.flush().map_err(PacaError::FileWrite)?;
    Ok(())
}

fn calculate_chunks(total_size: u64, count: usize) -> Vec<(u64, u64)> {
    let chunk_size = total_size / count as u64;
    (0..count)
        .map(|i| {
            let start = i as u64 * chunk_size;
            let end = if i == count - 1 {
                total_size - 1
            } else {
                (i as u64 + 1) * chunk_size - 1
            };
            (start, end)
        })
        .collect()
}

fn pending_style() -> ProgressStyle {
    ProgressStyle::default_bar().template("{msg}").unwrap()
}

fn download_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
        .unwrap()
        .progress_chars("#>-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_retryable_returns_true_for_rate_limited() {
        assert!(is_retryable(&PacaError::RateLimited(30)));
    }

    #[test]
    fn is_retryable_returns_true_for_rate_limited_without_retry_after() {
        assert!(is_retryable(&PacaError::RateLimited(0)));
    }

    #[test]
    fn calculate_chunks_divides_evenly() {
        let chunks = calculate_chunks(100, 4);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], (0, 24));
        assert_eq!(chunks[1], (25, 49));
        assert_eq!(chunks[2], (50, 74));
        assert_eq!(chunks[3], (75, 99));
    }

    #[test]
    fn calculate_chunks_last_chunk_absorbs_remainder() {
        let chunks = calculate_chunks(10, 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], (0, 2));
        assert_eq!(chunks[1], (3, 5));
        assert_eq!(chunks[2], (6, 9));
    }

    #[test]
    fn calculate_chunks_covers_entire_file() {
        let total_size = 1_048_576u64;
        let chunks = calculate_chunks(total_size, 4);
        let total_bytes: u64 = chunks.iter().map(|(start, end)| end - start + 1).sum();
        assert_eq!(total_bytes, total_size);
    }

    #[test]
    fn calculate_chunks_has_no_gaps() {
        let chunks = calculate_chunks(1000, 4);
        for i in 1..chunks.len() {
            assert_eq!(chunks[i].0, chunks[i - 1].1 + 1);
        }
    }

    #[tokio::test]
    async fn download_model_returns_error_for_missing_tag() {
        let result = download_model("owner/model", None).await;
        assert!(result.is_err());
    }

    #[test]
    fn create_snapshot_symlink_creates_symlink_for_root_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        let blob_dir = blobs_dir(dir.path(), &model_ref);
        fs::create_dir_all(&blob_dir).unwrap();
        fs::write(blob_dir.join("abc123hash"), b"fake data").unwrap();

        let result = create_snapshot_symlink(
            dir.path(),
            &model_ref,
            "commitabc",
            "model-Q4.gguf",
            "abc123hash",
        )
        .unwrap();

        assert!(result.symlink_metadata().unwrap().file_type().is_symlink());
        let target = fs::read_link(&result).unwrap();
        assert_eq!(target.to_str().unwrap(), "../../blobs/abc123hash");
    }

    #[test]
    fn create_snapshot_symlink_creates_symlink_for_subdir_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:BF16".parse().unwrap();
        let blob_dir = blobs_dir(dir.path(), &model_ref);
        fs::create_dir_all(&blob_dir).unwrap();
        fs::write(blob_dir.join("def456hash"), b"fake data").unwrap();

        let result = create_snapshot_symlink(
            dir.path(),
            &model_ref,
            "commitdef",
            "BF16/model-BF16-00001-of-00002.gguf",
            "def456hash",
        )
        .unwrap();

        assert!(result.symlink_metadata().unwrap().file_type().is_symlink());
        let target = fs::read_link(&result).unwrap();
        assert_eq!(target.to_str().unwrap(), "../../../blobs/def456hash");
    }

    #[test]
    fn create_snapshot_symlink_replaces_existing_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        let blob_dir = blobs_dir(dir.path(), &model_ref);
        fs::create_dir_all(&blob_dir).unwrap();
        fs::write(blob_dir.join("hash1"), b"data1").unwrap();
        fs::write(blob_dir.join("hash2"), b"data2").unwrap();

        create_snapshot_symlink(dir.path(), &model_ref, "commit1", "model.gguf", "hash1").unwrap();

        let result =
            create_snapshot_symlink(dir.path(), &model_ref, "commit1", "model.gguf", "hash2")
                .unwrap();

        let target = fs::read_link(&result).unwrap();
        assert_eq!(target.to_str().unwrap(), "../../blobs/hash2");
    }
}
