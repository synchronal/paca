pub use crate::error::PacaError;

use std::fs::{self, File};
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;

use crate::cache::{blob_exists, blobs_dir, get_hub_dir, save_ref, snapshots_dir};
use crate::model::ModelRef;
use crate::progress::FileProgress;
use crate::registry::default_headers;
use crate::registry::endpoint::get_model_endpoint;
use crate::registry::manifest::{GgufFile, fetch_manifest as fetch_registry_manifest};
use crate::registry::{fetch_resolve_info, resolve_client};
use crate::sysinfo::check_disk_space;

/// A prepared download manifest: the parsed model ref plus the GGUF
/// files that will be fetched. Returned by [`fetch_manifest`] so callers
/// can create one progress reporter per file before invoking
/// [`download_model`].
pub struct ModelManifest {
    model_ref: ModelRef,
    files: Vec<GgufFile>,
}

impl ModelManifest {
    /// Iterator over `(filename, size_in_bytes)` tuples for every GGUF
    /// file that will be downloaded.
    pub fn files(&self) -> impl ExactSizeIterator<Item = (&str, u64)> + '_ {
        self.files.iter().map(|f| (f.filename.as_str(), f.size))
    }
}

/// Fetches the model manifest from HuggingFace without starting the download.
pub async fn fetch_manifest(model: &str) -> Result<ModelManifest, PacaError> {
    let model_ref: ModelRef = model.parse()?;
    let client = build_download_client(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)?;
    let manifest = fetch_registry_manifest(&client, &model_ref).await?;
    Ok(ModelManifest {
        files: manifest.gguf_files,
        model_ref,
    })
}

/// Downloads a GGUF model from HuggingFace into the HF Hub cache format.
///
/// `progress` must contain one reporter per file in `manifest`, in the
/// same order as [`ModelManifest::files`].
pub async fn download_model(
    manifest: ModelManifest,
    hub_dir: Option<PathBuf>,
    progress: Vec<Arc<dyn FileProgress>>,
) -> Result<Vec<PathBuf>, PacaError> {
    let ModelManifest { files, model_ref } = manifest;
    assert_eq!(
        files.len(),
        progress.len(),
        "progress reporter count must match manifest file count"
    );

    let client = build_download_client(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)?;
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

    let total_size: u64 = files.iter().map(|f| f.size).sum();
    check_disk_space(&blobs, total_size)?;

    let mut set: tokio::task::JoinSet<Result<(PathBuf, String), PacaError>> =
        tokio::task::JoinSet::new();

    for (gguf_file, bar) in files.into_iter().zip(progress) {
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

                if blob_is_complete(existing_size, gguf_file.size) {
                    bar.start(gguf_file.size);
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

                // A final blob whose size doesn't match is evidence of a
                // legacy (pre-atomic-rename) download or external tampering.
                // Delete and redownload through the .partial + rename path.
                fs::remove_file(&blob_path).map_err(PacaError::FileDelete)?;
            }

            download_to_blob(&client, &url, &blob_path, gguf_file.size, &bar).await?;

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
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let mut retries: u32 = 0;
    let mut bytes_on_disk = resume_from;

    loop {
        match attempt_download(client, url, path, bytes_on_disk, progress).await {
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
                progress.println(&format!(
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
        PacaError::RangeNotHonored(_) => true,
        PacaError::RateLimited(_) => true,
        _ => false,
    }
}

async fn attempt_download(
    client: &Client,
    url: &str,
    path: &Path,
    resume_from: u64,
    progress: &Arc<dyn FileProgress>,
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

    progress.start(start_pos);

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PacaError::Download(std::io::Error::other(e)))?
    {
        file.write_all(&chunk).map_err(PacaError::FileWrite)?;
        progress.inc(chunk.len() as u64);
    }

    file.flush().map_err(PacaError::FileWrite)?;
    progress.finish();

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
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let chunks = calculate_chunks(total_size, CHUNK_COUNT);

    let file = File::create(path).map_err(PacaError::FileWrite)?;
    file.set_len(total_size).map_err(PacaError::FileWrite)?;
    drop(file);

    progress.start(0);

    let mut set = tokio::task::JoinSet::new();

    for (start, end) in chunks {
        let client = client.clone();
        let url = url.to_string();
        let path = path.to_path_buf();
        let bar = Arc::clone(progress);

        set.spawn(async move { download_chunk(&client, &url, &path, start, end, &bar).await });
    }

    while let Some(result) = set.join_next().await {
        result.expect("chunk download task panicked")?;
    }

    progress.finish();
    Ok(())
}

async fn download_chunk(
    client: &Client,
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let mut retries: u32 = 0;
    let mut bytes_written: u64 = 0;
    let chunk_size = end - start + 1;

    loop {
        let current_start = start + bytes_written;
        match attempt_chunk_download(client, url, path, current_start, end, progress).await {
            Ok(received) => {
                bytes_written += received;
                if bytes_written >= chunk_size {
                    return Ok(());
                }

                // Server closed early without delivering the full range.
                // Treat making any progress as a reset for the retry counter.
                if received > 0 {
                    retries = 0;
                } else {
                    retries += 1;
                }

                if retries > MAX_RETRIES {
                    return Err(PacaError::Download(std::io::Error::other(
                        "server closed connection before delivering the full chunk",
                    )));
                }

                let delay_secs = 1u64 << retries;
                progress.println(&format!(
                    "Chunk ended early ({bytes_written}/{chunk_size} bytes). Retrying in {delay_secs}s (attempt {retries}/{MAX_RETRIES})..."
                ));
                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            }
            Err(e) if is_retryable(&e) => {
                retries += 1;
                if retries > MAX_RETRIES {
                    return Err(e);
                }

                let delay_secs = match &e {
                    PacaError::RateLimited(wait) if *wait > 0 => *wait,
                    _ => 1u64 << retries,
                };
                progress.println(&format!(
                    "Chunk download error: {e}. Retrying in {delay_secs}s (attempt {retries}/{MAX_RETRIES})..."
                ));
                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn attempt_chunk_download(
    client: &Client,
    url: &str,
    path: &Path,
    start: u64,
    end: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<u64, PacaError> {
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

    let response = response.error_for_status()?;

    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(PacaError::RangeNotHonored(response.status().as_u16()));
    }

    let mut response = response;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(PacaError::FileWrite)?;
    file.seek(std::io::SeekFrom::Start(start))
        .map_err(PacaError::FileWrite)?;

    let mut bytes_received: u64 = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PacaError::Download(std::io::Error::other(e)))?
    {
        file.write_all(&chunk).map_err(PacaError::FileWrite)?;
        progress.inc(chunk.len() as u64);
        bytes_received += chunk.len() as u64;
    }

    file.flush().map_err(PacaError::FileWrite)?;
    Ok(bytes_received)
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

/// Connection establishment timeout for chunk downloads.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-read timeout — fires if the server goes silent mid-response,
/// which is the "hang" mode we've seen in the wild.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(60);

fn build_download_client(
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<Client, PacaError> {
    Ok(Client::builder()
        .connect_timeout(connect_timeout)
        .default_headers(default_headers())
        .read_timeout(read_timeout)
        .tcp_keepalive(Duration::from_secs(15))
        .build()?)
}

fn blob_is_complete(existing_size: u64, expected_size: u64) -> bool {
    existing_size == expected_size
}

fn partial_path(final_path: &Path) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    final_path.with_file_name(name)
}

/// Downloads `total_size` bytes from `url` into `<final_path>.partial`, then
/// atomically renames to `final_path` on success. An interrupted run leaves
/// only a `.partial` file — never a misleadingly-sized final blob.
async fn download_to_blob(
    client: &Client,
    url: &str,
    final_path: &Path,
    total_size: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let partial = partial_path(final_path);

    let existing = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    let resume_from = if existing > total_size {
        fs::remove_file(&partial).map_err(PacaError::FileDelete)?;
        0
    } else {
        existing
    };

    if resume_from == 0 && total_size >= PARALLEL_THRESHOLD {
        download_file_parallel(client, url, &partial, total_size, progress).await?;
    } else {
        download_file(client, url, &partial, resume_from, progress).await?;
    }

    verify_file_size(&partial, total_size)?;
    fs::rename(&partial, final_path).map_err(PacaError::FileWrite)?;
    Ok(())
}

fn verify_file_size(path: &Path, expected: u64) -> Result<(), PacaError> {
    let actual = fs::metadata(path).map_err(PacaError::FileWrite)?.len();
    if actual != expected {
        return Err(PacaError::SizeMismatch { actual, expected });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct NoopProgress;

    impl FileProgress for NoopProgress {
        fn start(&self, _: u64) {}
        fn inc(&self, _: u64) {}
        fn println(&self, _: &str) {}
        fn finish(&self) {}
    }

    fn noop_progress() -> Arc<dyn FileProgress> {
        Arc::new(NoopProgress)
    }

    fn preallocated_file(path: &Path, size: u64) {
        let file = File::create(path).unwrap();
        file.set_len(size).unwrap();
    }

    #[tokio::test]
    async fn attempt_chunk_download_errs_when_server_returns_200_to_range_request() {
        let server = MockServer::start().await;
        let body = vec![7u8; 64];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        preallocated_file(&path, 64);
        let progress = noop_progress();

        let result = attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress).await;

        assert!(
            matches!(result, Err(PacaError::RangeNotHonored(200))),
            "expected RangeNotHonored(200), got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn attempt_chunk_download_returns_bytes_received_on_206() {
        let server = MockServer::start().await;
        let body = vec![3u8; 32];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        preallocated_file(&path, 64);
        let progress = noop_progress();

        let bytes = attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress)
            .await
            .unwrap();

        assert_eq!(bytes, 32);
    }

    #[tokio::test]
    async fn is_retryable_returns_true_for_range_not_honored() {
        assert!(is_retryable(&PacaError::RangeNotHonored(200)));
    }

    #[test]
    fn partial_path_appends_partial_suffix() {
        let blob = PathBuf::from("/tmp/blobs/abc123");
        assert_eq!(
            partial_path(&blob),
            PathBuf::from("/tmp/blobs/abc123.partial")
        );
    }

    #[test]
    fn partial_path_preserves_hash_with_dots() {
        let blob = PathBuf::from("/tmp/blobs/abc.def");
        assert_eq!(
            partial_path(&blob),
            PathBuf::from("/tmp/blobs/abc.def.partial")
        );
    }

    #[tokio::test]
    async fn download_to_blob_writes_final_file_and_removes_partial() {
        let server = MockServer::start().await;
        let body = b"hello world".to_vec();
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        download_to_blob(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert!(final_path.exists(), "final blob should exist");
        assert!(
            !partial_path(&final_path).exists(),
            "partial file should have been renamed away"
        );
        assert_eq!(fs::read(&final_path).unwrap(), body);
    }

    #[tokio::test]
    async fn download_to_blob_cleans_up_oversized_partial_before_downloading() {
        let server = MockServer::start().await;
        let body = b"small body".to_vec();
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        // Simulate a corrupt prior run: oversized partial on disk.
        fs::write(partial_path(&final_path), vec![0u8; 9999]).unwrap();
        let progress = noop_progress();

        download_to_blob(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), body);
        assert!(!partial_path(&final_path).exists());
    }

    #[tokio::test]
    async fn attempt_chunk_download_times_out_when_server_stalls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(206)
                    .set_body_bytes(vec![0u8; 32])
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&server)
            .await;

        let client =
            build_download_client(Duration::from_secs(1), Duration::from_millis(200)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        preallocated_file(&path, 64);
        let progress = noop_progress();

        let started = std::time::Instant::now();
        let result = attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress).await;
        let elapsed = started.elapsed();

        assert!(result.is_err(), "expected error, got {:?}", result);
        assert!(
            elapsed < Duration::from_secs(2),
            "expected read timeout to fire well under the 5s server delay, took {:?}",
            elapsed
        );
    }

    #[test]
    fn blob_is_complete_returns_true_for_exact_size_match() {
        assert!(blob_is_complete(1024, 1024));
    }

    #[test]
    fn blob_is_complete_returns_false_for_undersized_blob() {
        assert!(!blob_is_complete(512, 1024));
    }

    #[test]
    fn blob_is_complete_returns_false_for_oversized_blob() {
        // Regression: previously this case was treated as "complete" because
        // the check used `>=`. An oversized blob is evidence of a corrupted
        // prior download and must be redownloaded, not trusted.
        assert!(!blob_is_complete(2048, 1024));
    }

    #[test]
    fn verify_file_size_returns_ok_when_size_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        fs::write(&path, vec![0u8; 128]).unwrap();

        assert!(verify_file_size(&path, 128).is_ok());
    }

    #[test]
    fn verify_file_size_returns_err_when_file_is_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        fs::write(&path, vec![0u8; 256]).unwrap();

        let err = verify_file_size(&path, 128).unwrap_err();
        assert!(matches!(
            err,
            PacaError::SizeMismatch {
                actual: 256,
                expected: 128
            }
        ));
    }

    #[test]
    fn verify_file_size_returns_err_when_file_is_undersized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        fs::write(&path, vec![0u8; 64]).unwrap();

        let err = verify_file_size(&path, 128).unwrap_err();
        assert!(matches!(
            err,
            PacaError::SizeMismatch {
                actual: 64,
                expected: 128
            }
        ));
    }

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
    async fn fetch_manifest_returns_error_for_missing_tag() {
        let result = fetch_manifest("owner/model").await;
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
