use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;

use crate::cache::{HubLayout, ModelPaths};
use crate::error::PacaError;
use crate::model::ModelRef;
use crate::progress::FileProgress;
use crate::registry::default_headers;
use crate::registry::endpoint::model_endpoint;
use crate::registry::manifest::{GgufFile, fetch_manifest as fetch_registry_manifest};
use crate::registry::{ResolveInfo, build_resolve_client, fetch_resolve_info};
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
    let hub = HubLayout::open(hub_dir)?;
    let endpoint = model_endpoint();
    let head_client = build_resolve_client()?;

    let blobs = hub.model(&model_ref).blobs();
    fs::create_dir_all(&blobs).map_err(PacaError::CacheDir)?;

    // Resolving every file up front is what makes the disk-space check
    // honest: only once the blob hashes are known can already-cached files
    // be excluded from the requirement.
    let resolved = resolve_files(&head_client, endpoint, &model_ref, files, progress).await?;
    check_disk_space(&blobs, bytes_to_download(&hub, &model_ref, &resolved))?;

    let mut set: tokio::task::JoinSet<Result<(PathBuf, String), PacaError>> =
        tokio::task::JoinSet::new();

    for file in resolved {
        let client = client.clone();
        let model_ref = model_ref.clone();
        let hub = hub.clone();

        set.spawn(async move { install_file(&client, &hub, &model_ref, file).await });
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
        hub.model(&model_ref).save_ref(commit)?;
    }

    Ok(paths)
}

/// A manifest file with its registry-resolved blob and commit hashes.
struct ResolvedFile {
    gguf_file: GgufFile,
    progress: Arc<dyn FileProgress>,
    resolve_info: ResolveInfo,
    url: String,
}

/// Issues the resolve HEAD request for every file concurrently. Results
/// come back in completion order, which matches how `download_model`
/// already collects its paths.
async fn resolve_files(
    head_client: &Client,
    endpoint: &str,
    model_ref: &ModelRef,
    files: Vec<GgufFile>,
    progress: Vec<Arc<dyn FileProgress>>,
) -> Result<Vec<ResolvedFile>, PacaError> {
    let mut set: tokio::task::JoinSet<Result<ResolvedFile, PacaError>> =
        tokio::task::JoinSet::new();

    for (gguf_file, bar) in files.into_iter().zip(progress) {
        let head_client = head_client.clone();
        let url = format!(
            "{endpoint}/{}/resolve/main/{}",
            model_ref.repo(),
            gguf_file.filename
        );

        set.spawn(async move {
            let resolve_info = fetch_resolve_info(&head_client, &url).await?;
            Ok(ResolvedFile {
                gguf_file,
                progress: bar,
                resolve_info,
                url,
            })
        });
    }

    let mut resolved = Vec::new();
    while let Some(result) = set.join_next().await {
        resolved.push(result.expect("resolve task panicked")?);
    }

    Ok(resolved)
}

/// Bytes that actually have to be fetched, skipping blobs already on disk
/// at their expected size. Counting those would make a re-run of an
/// already-complete download demand the model's full size in free space.
fn bytes_to_download(hub: &HubLayout, model_ref: &ModelRef, resolved: &[ResolvedFile]) -> u64 {
    let paths = hub.model(model_ref);

    resolved
        .iter()
        .filter(|file| {
            let existing = fs::metadata(paths.blob(&file.resolve_info.blob_hash))
                .map_or(0, |metadata| metadata.len());
            !blob_is_complete(existing, file.gguf_file.size)
        })
        .map(|file| file.gguf_file.size)
        .sum()
}

/// Puts one file into the cache: fetches its blob unless a complete copy
/// is already on disk, then links it into the snapshot tree. Returns the
/// symlink path and the commit it belongs to.
async fn install_file(
    client: &Client,
    hub: &HubLayout,
    model_ref: &ModelRef,
    file: ResolvedFile,
) -> Result<(PathBuf, String), PacaError> {
    let ResolvedFile {
        gguf_file,
        progress,
        resolve_info,
        url,
    } = file;

    let paths = hub.model(model_ref);
    let blob_path = paths.blob(&resolve_info.blob_hash);

    if paths.blob_exists(&resolve_info.blob_hash) {
        let existing_size = fs::metadata(&blob_path).map_or(0, |m| m.len());

        if blob_is_complete(existing_size, gguf_file.size) {
            progress.start(gguf_file.size);
            progress.finish();
        } else {
            // A final blob whose size doesn't match is evidence of a
            // legacy (pre-atomic-rename) download or external tampering.
            // Delete and redownload through the .partial + rename path.
            fs::remove_file(&blob_path).map_err(PacaError::FileDelete)?;
            download_to_blob(client, &url, &blob_path, gguf_file.size, &progress).await?;
        }
    } else {
        download_to_blob(client, &url, &blob_path, gguf_file.size, &progress).await?;
    }

    let symlink_path = create_snapshot_symlink(
        &paths,
        &resolve_info.commit_hash,
        &gguf_file.filename,
        &resolve_info.blob_hash,
    )?;
    Ok((symlink_path, resolve_info.commit_hash))
}

fn create_snapshot_symlink(
    paths: &ModelPaths<'_>,
    commit_hash: &str,
    filename: &str,
    blob_hash: &str,
) -> Result<PathBuf, PacaError> {
    let symlink_path = paths.snapshot(commit_hash).join(filename);
    if let Some(parent) = symlink_path.parent() {
        fs::create_dir_all(parent).map_err(PacaError::CacheDir)?;
    }

    // Depth: one `..` to escape the commit dir, one per subdir inside it,
    // and one more to exit `snapshots/`.
    let depth = filename.matches('/').count() + 2;
    let relative_blob = format!("{}blobs/{blob_hash}", "../".repeat(depth));

    if symlink_path.symlink_metadata().is_ok() {
        fs::remove_file(&symlink_path).map_err(PacaError::FileDelete)?;
    }

    std::os::unix::fs::symlink(&relative_blob, &symlink_path).map_err(PacaError::Symlink)?;

    Ok(symlink_path)
}

const MAX_RETRIES: u32 = 5;

/// Retries a whole-body GET into `path` until it completes, resuming from
/// `resume_from` and restarting the retry budget whenever an attempt makes
/// forward progress.
async fn download_with_resume(
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
                let new_size = fs::metadata(path).map_or(bytes_on_disk, |m| m.len());

                if new_size > bytes_on_disk {
                    retries = 0;
                    bytes_on_disk = new_size;
                } else {
                    retries += 1;
                }

                if retries > MAX_RETRIES {
                    return Err(e);
                }

                let delay = retry_delay(&e, retries);
                progress.println(&format!(
                    "Download error: {e}. Retrying in {}s (attempt {retries}/{MAX_RETRIES})...",
                    delay.as_secs()
                ));
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

fn is_retryable(error: &PacaError) -> bool {
    match error {
        PacaError::Download(_) | PacaError::RangeNotHonored(_) | PacaError::RateLimited(_) => true,
        PacaError::ManifestFetch(e) => e.status().is_none_or(|status| status.is_server_error()),
        _ => false,
    }
}

fn retry_delay(error: &PacaError, attempt: u32) -> Duration {
    match error {
        PacaError::RateLimited(wait) if *wait > 0 => Duration::from_secs(*wait),
        _ => Duration::from_secs(1u64 << attempt.min(30)),
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
        request = request.header("Range", format!("bytes={resume_from}-"));
    }

    let response = request.send().await?;

    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(PacaError::RateLimited(parse_retry_after(&response)));
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

fn parse_retry_after(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Minimum file size to use parallel chunk downloads (100 MB)
const PARALLEL_THRESHOLD: u64 = 100 * 1024 * 1024;

/// Number of concurrent connections per file
const CHUNK_COUNT: usize = 4;

/// Each chunk gets its own `<final>.partial.<idx>` file so resume can
/// trust file length — never preallocate, or the size will lie.
async fn download_blob_parallel(
    client: &Client,
    url: &str,
    final_path: &Path,
    total_size: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let chunks = chunk_ranges(total_size, CHUNK_COUNT);
    let chunk_paths: Vec<PathBuf> = (0..chunks.len())
        .map(|i| chunk_partial_path(final_path, i))
        .collect();

    // Chunk files are the source of truth in parallel mode; a leftover
    // merged partial from a prior crashed concat would shadow them.
    let merged = partial_path(final_path);
    if merged.exists() {
        fs::remove_file(&merged).map_err(PacaError::FileDelete)?;
    }

    progress.start(0);

    let mut set = tokio::task::JoinSet::new();
    for (i, &(start, end)) in chunks.iter().enumerate() {
        let chunk_size = end - start + 1;
        let path = chunk_paths[i].clone();
        let existing = fs::metadata(&path).map_or(0, |m| m.len());
        let resume_from = if existing > chunk_size {
            fs::remove_file(&path).map_err(PacaError::FileDelete)?;
            0
        } else {
            existing
        };
        progress.inc(resume_from);
        if resume_from >= chunk_size {
            continue;
        }
        let client = client.clone();
        let url = url.to_string();
        let bar = Arc::clone(progress);
        set.spawn(async move { download_chunk(&client, &url, &path, start, end, &bar).await });
    }

    while let Some(result) = set.join_next().await {
        result.expect("chunk download task panicked")?;
    }

    concatenate_chunks(&merged, &chunk_paths)?;
    verify_file_size(&merged, total_size)?;
    fs::rename(&merged, final_path).map_err(PacaError::FileWrite)?;

    progress.finish();
    Ok(())
}

async fn download_chunk(
    client: &Client,
    url: &str,
    chunk_path: &Path,
    abs_start: u64,
    abs_end: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let chunk_size = abs_end - abs_start + 1;
    let mut retries: u32 = 0;
    let mut last_size: u64 = fs::metadata(chunk_path).map_or(0, |m| m.len());

    loop {
        if last_size >= chunk_size {
            return Ok(());
        }
        let current_start = abs_start + last_size;
        let result =
            attempt_chunk_download(client, url, chunk_path, current_start, abs_end, progress).await;
        let new_size = fs::metadata(chunk_path).map_or(last_size, |m| m.len());

        match result {
            Ok(()) => {
                if new_size >= chunk_size {
                    return Ok(());
                }
                if new_size > last_size {
                    retries = 0;
                } else {
                    retries += 1;
                }
                if retries > MAX_RETRIES {
                    return Err(PacaError::Download(std::io::Error::other(
                        "server closed connection before delivering the full chunk",
                    )));
                }
                let delay = Duration::from_secs(1u64 << retries.min(30));
                progress.println(&format!(
                    "Chunk ended early ({new_size}/{chunk_size} bytes). Retrying in {}s (attempt {retries}/{MAX_RETRIES})...",
                    delay.as_secs()
                ));
                tokio::time::sleep(delay).await;
            }
            Err(e) if is_retryable(&e) => {
                if new_size > last_size {
                    retries = 0;
                } else {
                    retries += 1;
                }
                if retries > MAX_RETRIES {
                    return Err(e);
                }
                let delay = retry_delay(&e, retries);
                progress.println(&format!(
                    "Chunk download error: {e}. Retrying in {}s (attempt {retries}/{MAX_RETRIES})...",
                    delay.as_secs()
                ));
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }

        last_size = new_size;
    }
}

async fn attempt_chunk_download(
    client: &Client,
    url: &str,
    chunk_path: &Path,
    start: u64,
    end: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let response = client
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .await?;

    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(PacaError::RateLimited(parse_retry_after(&response)));
    }

    let mut response = response.error_for_status()?;

    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(PacaError::RangeNotHonored(response.status().as_u16()));
    }

    // Append + create: each retry resumes from the file's current end, so any
    // bytes already on disk from a prior attempt are preserved and never
    // re-requested.
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(chunk_path)
        .map_err(PacaError::FileWrite)?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| PacaError::Download(std::io::Error::other(e)))?
    {
        file.write_all(&chunk).map_err(PacaError::FileWrite)?;
        progress.inc(chunk.len() as u64);
    }

    file.flush().map_err(PacaError::FileWrite)?;
    Ok(())
}

/// Removes each chunk after copying it so peak disk usage during concat
/// stays near `total_size` instead of `2 * total_size`.
fn concatenate_chunks(output: &Path, chunk_paths: &[PathBuf]) -> Result<(), PacaError> {
    let mut writer = BufWriter::new(File::create(output).map_err(PacaError::FileWrite)?);
    for path in chunk_paths {
        let mut reader = File::open(path).map_err(PacaError::FileWrite)?;
        std::io::copy(&mut reader, &mut writer).map_err(PacaError::FileWrite)?;
        drop(reader);
        fs::remove_file(path).map_err(PacaError::FileDelete)?;
    }
    writer.flush().map_err(PacaError::FileWrite)?;
    Ok(())
}

fn chunk_ranges(total_size: u64, count: usize) -> Vec<(u64, u64)> {
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

fn chunk_partial_path(final_path: &Path, idx: usize) -> PathBuf {
    let mut name = final_path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".partial.{idx}"));
    final_path.with_file_name(name)
}

/// Writes `total_size` bytes from `url` to `final_path`. Crash-safe: an
/// interrupted run leaves resumable `.partial*` files, never a
/// misleadingly-sized final blob.
async fn download_to_blob(
    client: &Client,
    url: &str,
    final_path: &Path,
    total_size: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    if total_size >= PARALLEL_THRESHOLD {
        download_blob_parallel(client, url, final_path, total_size, progress).await
    } else {
        download_blob_sequential(client, url, final_path, total_size, progress).await
    }
}

async fn download_blob_sequential(
    client: &Client,
    url: &str,
    final_path: &Path,
    total_size: u64,
    progress: &Arc<dyn FileProgress>,
) -> Result<(), PacaError> {
    let partial = partial_path(final_path);

    let existing = fs::metadata(&partial).map_or(0, |m| m.len());
    let resume_from = if existing > total_size {
        fs::remove_file(&partial).map_err(PacaError::FileDelete)?;
        0
    } else {
        existing
    };

    download_with_resume(client, url, &partial, resume_from, progress).await?;
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
        let progress = noop_progress();

        let result = attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress).await;

        assert!(
            matches!(result, Err(PacaError::RangeNotHonored(200))),
            "expected RangeNotHonored(200), got {result:?}"
        );
    }

    #[tokio::test]
    async fn attempt_chunk_download_appends_body_to_chunk_file_on_206() {
        let server = MockServer::start().await;
        let body = vec![3u8; 32];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.partial.0");
        let progress = noop_progress();

        attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress)
            .await
            .unwrap();

        assert_eq!(fs::read(&path).unwrap(), body);
    }

    #[tokio::test]
    async fn attempt_chunk_download_appends_to_existing_chunk_file() {
        let server = MockServer::start().await;
        let body = vec![3u8; 16];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.partial.0");
        fs::write(&path, vec![1u8; 16]).unwrap();
        let progress = noop_progress();

        attempt_chunk_download(&client, &server.uri(), &path, 16, 31, &progress)
            .await
            .unwrap();

        let mut expected = vec![1u8; 16];
        expected.extend_from_slice(&body);
        assert_eq!(fs::read(&path).unwrap(), expected);
    }

    fn resolved_file(filename: &str, size: u64, blob_hash: &str) -> ResolvedFile {
        ResolvedFile {
            gguf_file: GgufFile {
                filename: filename.to_string(),
                size,
            },
            progress: noop_progress(),
            resolve_info: ResolveInfo {
                blob_hash: blob_hash.to_string(),
                commit_hash: "commit1".to_string(),
            },
            url: format!("http://example.test/{filename}"),
        }
    }

    #[test]
    fn bytes_to_download_excludes_complete_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        fs::create_dir_all(hub.model(&mr).blobs()).unwrap();
        fs::write(hub.model(&mr).blob("cached"), vec![0u8; 100]).unwrap();

        let resolved = vec![
            resolved_file("a.gguf", 100, "cached"),
            resolved_file("b.gguf", 250, "missing"),
        ];

        assert_eq!(bytes_to_download(&hub, &mr, &resolved), 250);
    }

    #[test]
    fn bytes_to_download_counts_blob_with_wrong_size() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        fs::create_dir_all(hub.model(&mr).blobs()).unwrap();
        fs::write(hub.model(&mr).blob("truncated"), vec![0u8; 50]).unwrap();

        let resolved = vec![resolved_file("a.gguf", 100, "truncated")];

        assert_eq!(bytes_to_download(&hub, &mr, &resolved), 100);
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
        let path = dir.path().join("blob.partial.0");
        let progress = noop_progress();

        let started = std::time::Instant::now();
        let result = attempt_chunk_download(&client, &server.uri(), &path, 0, 31, &progress).await;
        let elapsed = started.elapsed();

        assert!(result.is_err(), "expected error, got {result:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "expected read timeout to fire well under the 5s server delay, took {elapsed:?}"
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
    fn chunk_ranges_divides_evenly() {
        let chunks = chunk_ranges(100, 4);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], (0, 24));
        assert_eq!(chunks[1], (25, 49));
        assert_eq!(chunks[2], (50, 74));
        assert_eq!(chunks[3], (75, 99));
    }

    #[test]
    fn chunk_ranges_last_chunk_absorbs_remainder() {
        let chunks = chunk_ranges(10, 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], (0, 2));
        assert_eq!(chunks[1], (3, 5));
        assert_eq!(chunks[2], (6, 9));
    }

    #[test]
    fn chunk_ranges_covers_entire_file() {
        let total_size = 1_048_576u64;
        let chunks = chunk_ranges(total_size, 4);
        let total_bytes: u64 = chunks.iter().map(|(start, end)| end - start + 1).sum();
        assert_eq!(total_bytes, total_size);
    }

    #[test]
    fn chunk_ranges_has_no_gaps() {
        let chunks = chunk_ranges(1000, 4);
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
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        let paths = hub.model(&mr);
        fs::create_dir_all(paths.blobs()).unwrap();
        fs::write(paths.blob("abc123hash"), b"fake data").unwrap();

        let result =
            create_snapshot_symlink(&paths, "commitabc", "model-Q4.gguf", "abc123hash").unwrap();

        assert!(result.symlink_metadata().unwrap().file_type().is_symlink());
        let target = fs::read_link(&result).unwrap();
        assert_eq!(target.to_str().unwrap(), "../../blobs/abc123hash");
    }

    #[test]
    fn create_snapshot_symlink_creates_symlink_for_subdir_file() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr: ModelRef = "owner/model-GGUF:BF16".parse().unwrap();
        let paths = hub.model(&mr);
        fs::create_dir_all(paths.blobs()).unwrap();
        fs::write(paths.blob("def456hash"), b"fake data").unwrap();

        let result = create_snapshot_symlink(
            &paths,
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
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        let paths = hub.model(&mr);
        fs::create_dir_all(paths.blobs()).unwrap();
        fs::write(paths.blob("hash1"), b"data1").unwrap();
        fs::write(paths.blob("hash2"), b"data2").unwrap();

        create_snapshot_symlink(&paths, "commit1", "model.gguf", "hash1").unwrap();

        let result = create_snapshot_symlink(&paths, "commit1", "model.gguf", "hash2").unwrap();

        let target = fs::read_link(&result).unwrap();
        assert_eq!(target.to_str().unwrap(), "../../blobs/hash2");
    }

    #[test]
    fn chunk_partial_path_appends_index_suffix() {
        let blob = PathBuf::from("/tmp/blobs/abc123");
        assert_eq!(
            chunk_partial_path(&blob, 0),
            PathBuf::from("/tmp/blobs/abc123.partial.0")
        );
        assert_eq!(
            chunk_partial_path(&blob, 7),
            PathBuf::from("/tmp/blobs/abc123.partial.7")
        );
    }

    #[test]
    fn chunk_partial_path_preserves_dots_in_blob_name() {
        let blob = PathBuf::from("/tmp/blobs/abc.def");
        assert_eq!(
            chunk_partial_path(&blob, 2),
            PathBuf::from("/tmp/blobs/abc.def.partial.2")
        );
    }

    fn range_responder(body: Vec<u8>) -> impl Fn(&wiremock::Request) -> ResponseTemplate {
        move |req: &wiremock::Request| {
            let body_len = body.len();
            let range = req.headers.get("range").and_then(|v| v.to_str().ok());
            match range {
                Some(spec) if spec.starts_with("bytes=") => {
                    let r = &spec["bytes=".len()..];
                    let mut parts = r.splitn(2, '-');
                    let start: usize = parts.next().unwrap().parse().unwrap();
                    let end_str = parts.next().unwrap();
                    let end: usize = if end_str.is_empty() {
                        body_len.saturating_sub(1)
                    } else {
                        end_str.parse().unwrap()
                    };
                    if start >= body_len {
                        return ResponseTemplate::new(416);
                    }
                    let end_clamped = end.min(body_len - 1);
                    let slice = body[start..=end_clamped].to_vec();
                    ResponseTemplate::new(206).set_body_bytes(slice)
                }
                _ => ResponseTemplate::new(200).set_body_bytes(body.clone()),
            }
        }
    }

    #[tokio::test]
    async fn download_blob_parallel_writes_final_blob_and_removes_chunk_partials_on_success() {
        let body: Vec<u8> = (0..64u8).collect();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(range_responder(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        download_blob_parallel(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert!(final_path.exists(), "final blob should exist");
        assert_eq!(fs::read(&final_path).unwrap(), body);
        for i in 0..CHUNK_COUNT {
            let chunk = chunk_partial_path(&final_path, i);
            assert!(!chunk.exists(), "chunk partial {chunk:?} should be removed");
        }
        assert!(
            !partial_path(&final_path).exists(),
            "merged partial should be removed"
        );
    }

    #[tokio::test]
    async fn download_blob_parallel_skips_complete_chunk_partial() {
        let body: Vec<u8> = (0..64u8).collect();
        let chunks = chunk_ranges(body.len() as u64, CHUNK_COUNT);
        let (start0, end0) = chunks[0];

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(range_responder(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        let chunk0_path = chunk_partial_path(&final_path, 0);
        fs::write(&chunk0_path, &body[start0 as usize..=end0 as usize]).unwrap();

        download_blob_parallel(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), body);

        let received = server.received_requests().await.unwrap();
        let chunk0_range_prefix = format!("bytes={start0}-");
        let requested_chunk0 = received.iter().any(|r| {
            r.headers
                .get("range")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|s| s.starts_with(&chunk0_range_prefix))
        });
        assert!(
            !requested_chunk0,
            "chunk 0 was already complete; should not be requested again"
        );
    }

    #[tokio::test]
    async fn download_blob_parallel_resumes_undersized_chunk_partial() {
        let body: Vec<u8> = (0..64u8).collect();
        let chunks = chunk_ranges(body.len() as u64, CHUNK_COUNT);
        let (start0, end0) = chunks[0];
        let chunk0_size = (end0 - start0 + 1) as usize;
        let half = chunk0_size / 2;

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(range_responder(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        let chunk0_path = chunk_partial_path(&final_path, 0);
        fs::write(&chunk0_path, &body[start0 as usize..start0 as usize + half]).unwrap();

        download_blob_parallel(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), body);

        let received = server.received_requests().await.unwrap();
        let resume_start = start0 + half as u64;
        let expected_resume = format!("bytes={resume_start}-{end0}");
        let saw_resume = received.iter().any(|r| {
            r.headers.get("range").and_then(|v| v.to_str().ok()) == Some(expected_resume.as_str())
        });
        assert!(
            saw_resume,
            "expected a resume range request {expected_resume:?}, saw: {:?}",
            received
                .iter()
                .filter_map(|r| r.headers.get("range").and_then(|v| v.to_str().ok()))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn download_blob_parallel_discards_oversized_chunk_partial() {
        let body: Vec<u8> = (0..64u8).collect();
        let chunks = chunk_ranges(body.len() as u64, CHUNK_COUNT);
        let (start0, end0) = chunks[0];
        let chunk0_size = (end0 - start0 + 1) as usize;

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(range_responder(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        let chunk0_path = chunk_partial_path(&final_path, 0);
        fs::write(&chunk0_path, vec![0xFFu8; chunk0_size * 2]).unwrap();

        download_blob_parallel(
            &client,
            &server.uri(),
            &final_path,
            body.len() as u64,
            &progress,
        )
        .await
        .unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), body);
    }

    #[tokio::test]
    async fn download_blob_parallel_clears_stale_merged_partial() {
        let body: Vec<u8> = (0..64u8).collect();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(range_responder(body.clone()))
            .mount(&server)
            .await;

        let client = Client::new();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("abc123");
        let progress = noop_progress();

        // A previous run crashed mid-concat, leaving a garbage merged partial.
        fs::write(partial_path(&final_path), vec![0xAAu8; 999]).unwrap();

        download_blob_parallel(
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
}
