pub mod clean;

pub use crate::error::PacaError;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use reqwest::Client;

use crate::model::ModelRef;
use crate::registry::default_headers;
use crate::registry::endpoint::get_model_endpoint;
use crate::registry::manifest::fetch_manifest;
use crate::registry::{fetch_resolve_info, resolve_client};

/// Information about a downloaded model
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ModelInfo {
    /// The model reference (owner/model:tag)
    pub model_ref: ModelRef,
    /// Whether the model is installed
    pub installed: bool,
}

/// Information about a model with an outdated commit
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OutdatedModelInfo {
    /// The model reference (owner/model:tag)
    pub model_ref: ModelRef,
    /// The filename of the outdated file
    pub filename: String,
    /// The local file path
    pub file_path: PathBuf,
}

/// Returns the HuggingFace Hub cache directory (~/.cache/huggingface/hub)
pub(crate) fn get_hub_dir() -> Result<PathBuf, PacaError> {
    let hub_dir = dirs::home_dir()
        .ok_or_else(|| {
            PacaError::CacheDir(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))
        })?
        .join(".cache")
        .join("huggingface")
        .join("hub");

    fs::create_dir_all(&hub_dir).map_err(PacaError::CacheDir)?;

    Ok(hub_dir)
}

/// Returns the model directory name in HF Hub format: "models--{owner}--{model}"
pub(crate) fn model_dir_name(model_ref: &ModelRef) -> String {
    format!("models--{}--{}", model_ref.owner, model_ref.model)
}

/// Returns the blobs directory for a model
pub(crate) fn blobs_dir(hub_dir: &Path, model_ref: &ModelRef) -> PathBuf {
    hub_dir.join(model_dir_name(model_ref)).join("blobs")
}

/// Returns the refs directory for a model
pub(crate) fn refs_dir(hub_dir: &Path, model_ref: &ModelRef) -> PathBuf {
    hub_dir.join(model_dir_name(model_ref)).join("refs")
}

/// Returns the snapshots directory for a model
pub(crate) fn snapshots_dir(hub_dir: &Path, model_ref: &ModelRef) -> PathBuf {
    hub_dir.join(model_dir_name(model_ref)).join("snapshots")
}

/// Saves the commit hash to refs/main
pub(crate) fn save_ref(
    hub_dir: &Path,
    model_ref: &ModelRef,
    commit_hash: &str,
) -> Result<(), PacaError> {
    let dir = refs_dir(hub_dir, model_ref);
    fs::create_dir_all(&dir).map_err(PacaError::CacheDir)?;
    fs::write(dir.join("main"), commit_hash).map_err(PacaError::FileWrite)?;
    Ok(())
}

/// Reads the commit hash from refs/main
pub(crate) fn read_ref(hub_dir: &Path, model_ref: &ModelRef) -> Option<String> {
    let path = refs_dir(hub_dir, model_ref).join("main");
    fs::read_to_string(path).ok()
}

/// Checks whether a blob with the given hash exists
pub(crate) fn blob_exists(hub_dir: &Path, model_ref: &ModelRef, blob_hash: &str) -> bool {
    blobs_dir(hub_dir, model_ref).join(blob_hash).exists()
}

/// Lists all downloaded models from the hub directory
pub fn list_models(hub_dir: Option<PathBuf>) -> Result<Vec<ModelInfo>, PacaError> {
    let hub_dir = match hub_dir {
        Some(dir) => dir,
        None => get_hub_dir()?,
    };

    if !hub_dir.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    for entry in fs::read_dir(&hub_dir)? {
        let entry = entry?;
        let dir_name = entry.file_name().to_string_lossy().to_string();

        if !dir_name.starts_with("models--") || !entry.path().is_dir() {
            continue;
        }

        list_snapshot_models(&dir_name, &entry.path(), &mut models)?;
    }

    models.sort_by_key(|a| a.model_ref.to_string());

    Ok(models)
}

fn list_snapshot_models(
    dir_name: &str,
    model_dir: &Path,
    models: &mut Vec<ModelInfo>,
) -> Result<(), PacaError> {
    let (owner, model) = match parse_model_dir_name(dir_name) {
        Some(pair) => pair,
        None => return Ok(()),
    };

    // Find the current snapshot via refs/main
    let refs_main = model_dir.join("refs").join("main");
    let commit = match fs::read_to_string(refs_main) {
        Ok(c) => c.trim().to_string(),
        Err(_) => return Ok(()),
    };

    let snapshot_dir = model_dir.join("snapshots").join(&commit);
    if !snapshot_dir.is_dir() {
        return Ok(());
    }

    for tag in collect_gguf_tags(&snapshot_dir, &model)? {
        models.push(ModelInfo {
            model_ref: ModelRef {
                model: model.clone(),
                owner: owner.clone(),
                tag,
            },
            installed: true,
        });
    }

    Ok(())
}

fn collect_gguf_tags(dir: &Path, model: &str) -> Result<Vec<String>, PacaError> {
    let mut tags = Vec::new();
    collect_gguf_tags_recursive(dir, dir, model, &mut tags)?;
    tags.sort();
    tags.dedup();
    Ok(tags)
}

fn collect_gguf_tags_recursive(
    base: &Path,
    dir: &Path,
    model: &str,
    tags: &mut Vec<String>,
) -> Result<(), PacaError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_gguf_tags_recursive(base, &path, model, tags)?;
        } else {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".gguf") {
                let relative = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                let tag = if let Some((subdir, _)) = relative.split_once('/') {
                    subdir.to_string()
                } else if let Some(derived) = derive_tag(&name, model) {
                    derived
                } else {
                    continue;
                };

                tags.push(tag);
            }
        }
    }
    Ok(())
}

/// Derives a quantization tag from a GGUF filename using the model name.
///
/// Strips the model base name prefix (model name without `-GGUF`), the `.gguf`
/// suffix, and any shard suffix (e.g. `-00001-of-00002`).
fn derive_tag(filename: &str, model: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf")?;
    let model_base = model.strip_suffix("-GGUF")?;
    let remainder = stem.strip_prefix(model_base)?.strip_prefix('-')?;

    if remainder.is_empty() {
        return None;
    }

    // Strip shard suffix like -00001-of-00002
    let tag = if let Some((before, _)) = remainder.rsplit_once("-of-") {
        before.rsplit_once('-').map(|(t, _)| t).unwrap_or(before)
    } else {
        remainder
    };

    if tag.is_empty() {
        return None;
    }

    Some(tag.to_string())
}

fn parse_model_dir_name(dir_name: &str) -> Option<(String, String)> {
    let stripped = dir_name.strip_prefix("models--")?;
    let (owner, model) = stripped.split_once("--")?;
    Some((owner.to_string(), model.to_string()))
}

/// Checks which downloaded models have outdated files by comparing commit hashes.
///
/// Groups models by repo so that only one resolve-info HEAD request is made per
/// repo, regardless of how many tags are installed.
pub async fn check_outdated_models(
    hub_dir: Option<PathBuf>,
) -> Result<Vec<OutdatedModelInfo>, PacaError> {
    let hub_dir = match hub_dir {
        Some(dir) => dir,
        None => get_hub_dir()?,
    };

    if !hub_dir.exists() {
        return Ok(Vec::new());
    }

    let client = Client::builder()
        .default_headers(default_headers())
        .build()?;
    let head_client = resolve_client()?;
    let endpoint = get_model_endpoint();
    let mut outdated_models = Vec::new();

    let models = list_models(Some(hub_dir.clone()))?;

    // Track which repos have been checked: true = outdated, false = up to date
    let mut repo_outdated: HashMap<String, bool> = HashMap::new();

    for model_info in &models {
        let model_ref = &model_info.model_ref;
        let repo = model_ref.repo();

        let is_outdated = match repo_outdated.get(&repo) {
            Some(&val) => val,
            None => {
                let outdated =
                    check_repo_outdated(&client, &head_client, &endpoint, &hub_dir, model_ref)
                        .await?;
                repo_outdated.insert(repo, outdated);
                outdated
            }
        };

        if !is_outdated {
            continue;
        }

        let manifest = fetch_manifest(&client, model_ref).await?;
        let local_commit = read_ref(&hub_dir, model_ref);
        let snapshot_path =
            snapshots_dir(&hub_dir, model_ref).join(local_commit.as_deref().unwrap_or(""));

        for gguf_file in &manifest.gguf_files {
            outdated_models.push(OutdatedModelInfo {
                model_ref: model_ref.clone(),
                filename: gguf_file.filename.clone(),
                file_path: snapshot_path.join(&gguf_file.filename),
            });
        }
    }

    outdated_models.sort_by_key(|a| a.model_ref.to_string());

    Ok(outdated_models)
}

async fn check_repo_outdated(
    client: &Client,
    head_client: &Client,
    endpoint: &str,
    hub_dir: &Path,
    model_ref: &ModelRef,
) -> Result<bool, PacaError> {
    let manifest = fetch_manifest(client, model_ref).await?;
    let first_file = match manifest.gguf_files.first() {
        Some(f) => f,
        None => return Ok(false),
    };

    let url = format!(
        "{}/{}/resolve/main/{}",
        endpoint,
        model_ref.repo(),
        first_file.filename
    );

    let local_commit = read_ref(hub_dir, model_ref);

    match fetch_resolve_info(head_client, &url).await {
        Ok(info) => Ok(local_commit.as_deref() != Some(&info.commit_hash)),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_hub_dir_returns_huggingface_hub_subdirectory() {
        let hub_dir = get_hub_dir().unwrap();
        assert!(hub_dir.ends_with(".cache/huggingface/hub"));
    }

    #[test]
    fn model_dir_name_formats_correctly() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        assert_eq!(
            model_dir_name(&model_ref),
            "models--unsloth--GLM-4.7-Flash-GGUF"
        );
    }

    #[test]
    fn blobs_dir_returns_correct_path() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        let hub = PathBuf::from("/cache/huggingface/hub");
        assert_eq!(
            blobs_dir(&hub, &model_ref),
            PathBuf::from("/cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/blobs")
        );
    }

    #[test]
    fn refs_dir_returns_correct_path() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        let hub = PathBuf::from("/cache/huggingface/hub");
        assert_eq!(
            refs_dir(&hub, &model_ref),
            PathBuf::from("/cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/refs")
        );
    }

    #[test]
    fn snapshots_dir_returns_correct_path() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL".parse().unwrap();
        let hub = PathBuf::from("/cache/huggingface/hub");
        assert_eq!(
            snapshots_dir(&hub, &model_ref),
            PathBuf::from("/cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots")
        );
    }

    #[test]
    fn save_and_read_ref() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();

        save_ref(dir.path(), &model_ref, "abc123commit").unwrap();
        assert_eq!(
            read_ref(dir.path(), &model_ref),
            Some("abc123commit".to_string())
        );
    }

    #[test]
    fn read_ref_returns_none_when_no_ref_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        assert_eq!(read_ref(dir.path(), &model_ref), None);
    }

    #[test]
    fn blob_exists_returns_true_when_blob_exists() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();
        let blob_dir = blobs_dir(dir.path(), &model_ref);
        fs::create_dir_all(&blob_dir).unwrap();
        fs::write(blob_dir.join("abcdef1234"), b"data").unwrap();

        assert!(blob_exists(dir.path(), &model_ref, "abcdef1234"));
    }

    #[test]
    fn blob_exists_returns_false_when_blob_missing() {
        let dir = tempfile::tempdir().unwrap();
        let model_ref: ModelRef = "owner/model-GGUF:Q4".parse().unwrap();

        assert!(!blob_exists(dir.path(), &model_ref, "abcdef1234"));
    }

    #[test]
    fn list_models_returns_empty_for_nonexistent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does-not-exist");
        let result = list_models(Some(nonexistent)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn list_models_returns_empty_for_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = list_models(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn list_models_skips_non_model_directories() {
        let dir = tempfile::tempdir().unwrap();

        // Not a models-- directory
        let other_dir = dir.path().join("something-else").join("paca");
        fs::create_dir_all(&other_dir).unwrap();
        fs::write(other_dir.join("Q4.json"), r#"{"test": true}"#).unwrap();

        let result = list_models(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn derive_tag_strips_model_base_and_gguf_suffix() {
        assert_eq!(
            derive_tag(
                "Phi-4-mini-reasoning-BF16.gguf",
                "Phi-4-mini-reasoning-GGUF"
            ),
            Some("BF16".to_string())
        );
    }

    #[test]
    fn derive_tag_handles_quantization_with_underscores() {
        assert_eq!(
            derive_tag("Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-1.7B-GGUF"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn derive_tag_handles_ud_prefix_quantization() {
        assert_eq!(
            derive_tag("Qwen3.5-0.8B-UD-Q2_K_XL.gguf", "Qwen3.5-0.8B-GGUF"),
            Some("UD-Q2_K_XL".to_string())
        );
    }

    #[test]
    fn derive_tag_strips_shard_suffix() {
        assert_eq!(
            derive_tag("Model-BF16-00001-of-00002.gguf", "Model-GGUF"),
            Some("BF16".to_string())
        );
    }

    #[test]
    fn derive_tag_returns_none_for_non_matching_prefix() {
        assert_eq!(derive_tag("mmproj-BF16.gguf", "Qwen3.5-0.8B-GGUF"), None);
    }

    #[test]
    fn derive_tag_returns_none_for_non_gguf_file() {
        assert_eq!(derive_tag("readme.txt", "Model-GGUF"), None);
    }

    #[test]
    fn list_models_discovers_models_from_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("models--owner--model-GGUF");

        let refs = model_dir.join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), "abc123commit").unwrap();

        let snapshot = model_dir.join("snapshots").join("abc123commit");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model-Q4.gguf"), b"fake").unwrap();

        let result = list_models(Some(dir.path().to_path_buf())).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].model_ref.owner, "owner");
        assert_eq!(result[0].model_ref.model, "model-GGUF");
        assert_eq!(result[0].model_ref.tag, "Q4");
    }

    #[test]
    fn list_models_discovers_multiple_tags_from_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("models--owner--model-GGUF");

        let refs = model_dir.join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), "commit1").unwrap();

        let snapshot = model_dir.join("snapshots").join("commit1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model-Q4_K_M.gguf"), b"fake").unwrap();

        let bf16_dir = snapshot.join("BF16");
        fs::create_dir_all(&bf16_dir).unwrap();
        fs::write(bf16_dir.join("model-BF16-00001-of-00002.gguf"), b"fake").unwrap();

        let result = list_models(Some(dir.path().to_path_buf())).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].model_ref.tag, "BF16");
        assert_eq!(result[1].model_ref.tag, "Q4_K_M");
    }

    #[test]
    fn list_models_discovers_subdir_gguf_from_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("models--owner--model-GGUF");

        let refs = model_dir.join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), "commit1").unwrap();

        let snapshot = model_dir.join("snapshots").join("commit1").join("BF16");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model-BF16-00001-of-00002.gguf"), b"fake").unwrap();
        fs::write(snapshot.join("model-BF16-00002-of-00002.gguf"), b"fake").unwrap();

        let result = list_models(Some(dir.path().to_path_buf())).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].model_ref.tag, "BF16");
    }
}
