pub mod clean;
pub mod remove;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use reqwest::Client;

use crate::error::PacaError;
use crate::model::ModelRef;
use crate::registry::endpoint::model_endpoint;
use crate::registry::manifest::fetch_manifest;
use crate::registry::{default_headers, fetch_resolve_info, resolve_client};

/// Information about a model with an outdated commit
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct OutdatedModelInfo {
    pub model_ref: ModelRef,
    pub filename: String,
    pub file_path: PathBuf,
}

/// The HuggingFace Hub cache root, ensured to exist on disk.
///
/// Constructed from an optional user override via [`HubLayout::open`]; all
/// path derivations inside the cache go through this type rather than
/// threading `(hub_dir, model_ref)` pairs.
#[derive(Clone, Debug)]
pub(crate) struct HubLayout {
    root: PathBuf,
}

impl HubLayout {
    /// Opens the hub directory, creating it if missing.
    ///
    /// `override_path` replaces the default `~/.cache/huggingface/hub`
    /// location when set.
    pub(crate) fn open(override_path: Option<PathBuf>) -> Result<Self, PacaError> {
        let root = match override_path {
            Some(root) => root,
            None => default_hub_dir()?,
        };
        fs::create_dir_all(&root).map_err(PacaError::CacheDir)?;
        Ok(Self { root })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn model<'a>(&'a self, model_ref: &'a ModelRef) -> ModelPaths<'a> {
        ModelPaths {
            hub: &self.root,
            model_ref,
        }
    }
}

/// A view over the on-disk layout for a single model reference.
pub(crate) struct ModelPaths<'a> {
    hub: &'a Path,
    model_ref: &'a ModelRef,
}

impl ModelPaths<'_> {
    pub(crate) fn dir(&self) -> PathBuf {
        self.hub.join(model_dir_name(self.model_ref))
    }

    pub(crate) fn blobs(&self) -> PathBuf {
        self.dir().join("blobs")
    }

    pub(crate) fn refs(&self) -> PathBuf {
        self.dir().join("refs")
    }

    pub(crate) fn snapshots(&self) -> PathBuf {
        self.dir().join("snapshots")
    }

    pub(crate) fn ref_main(&self) -> PathBuf {
        self.refs().join("main")
    }

    pub(crate) fn blob(&self, hash: &str) -> PathBuf {
        self.blobs().join(hash)
    }

    pub(crate) fn snapshot(&self, commit_hash: &str) -> PathBuf {
        self.snapshots().join(commit_hash)
    }

    pub(crate) fn save_ref(&self, commit_hash: &str) -> Result<(), PacaError> {
        let dir = self.refs();
        fs::create_dir_all(&dir).map_err(PacaError::CacheDir)?;
        fs::write(dir.join("main"), commit_hash).map_err(PacaError::FileWrite)?;
        Ok(())
    }

    pub(crate) fn read_ref(&self) -> Option<String> {
        fs::read_to_string(self.ref_main()).ok()
    }

    pub(crate) fn blob_exists(&self, blob_hash: &str) -> bool {
        self.blob(blob_hash).exists()
    }
}

fn default_hub_dir() -> Result<PathBuf, PacaError> {
    let home = dirs::home_dir().ok_or_else(|| {
        PacaError::CacheDir(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Could not determine home directory",
        ))
    })?;
    Ok(home.join(".cache").join("huggingface").join("hub"))
}

pub(crate) fn model_dir_name(model_ref: &ModelRef) -> String {
    format!("models--{}--{}", model_ref.owner, model_ref.model)
}

/// Lists all downloaded models from the hub directory.
pub fn list_models(hub_dir: Option<PathBuf>) -> Result<Vec<ModelRef>, PacaError> {
    let hub = HubLayout::open(hub_dir)?;
    let mut models = Vec::new();

    for entry in fs::read_dir(hub.root()).map_err(PacaError::CacheDir)? {
        let entry = entry.map_err(PacaError::CacheDir)?;
        let dir_name = entry.file_name().to_string_lossy().into_owned();

        if !dir_name.starts_with("models--") || !entry.path().is_dir() {
            continue;
        }

        collect_snapshot_models(&dir_name, &entry.path(), &mut models)?;
    }

    models.sort_by_key(ModelRef::to_string);

    Ok(models)
}

fn collect_snapshot_models(
    dir_name: &str,
    model_dir: &Path,
    models: &mut Vec<ModelRef>,
) -> Result<(), PacaError> {
    let Some((owner, model)) = parse_model_dir_name(dir_name) else {
        return Ok(());
    };

    let Ok(commit) = fs::read_to_string(model_dir.join("refs").join("main")) else {
        return Ok(());
    };
    let commit = commit.trim();

    let snapshot_dir = model_dir.join("snapshots").join(commit);
    if !snapshot_dir.is_dir() {
        return Ok(());
    }

    for tag in collect_gguf_tags(&snapshot_dir, &model)? {
        models.push(ModelRef {
            model: model.clone(),
            owner: owner.clone(),
            tag,
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
    for entry in fs::read_dir(dir).map_err(PacaError::CacheDir)? {
        let entry = entry.map_err(PacaError::CacheDir)?;
        let path = entry.path();

        if path.is_dir() {
            collect_gguf_tags_recursive(base, &path, model, tags)?;
            continue;
        }

        if !is_gguf(&path) {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();

        if let Some((subdir, _)) = relative.split_once('/') {
            tags.push(subdir.to_string());
        } else if let Some(derived) = derive_tag(&name, model) {
            tags.push(derived);
        }
    }
    Ok(())
}

pub(crate) fn is_gguf(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

/// Derives a quantization tag from a GGUF filename using the model name.
///
/// Strips the model base name prefix (model name without `-GGUF`), the
/// `.gguf` suffix, and any shard suffix (e.g. `-00001-of-00002`).
pub(crate) fn derive_tag(filename: &str, model: &str) -> Option<String> {
    let stem = filename.strip_suffix(".gguf")?;
    let model_base = model.strip_suffix("-GGUF")?;
    let remainder = stem.strip_prefix(model_base)?.strip_prefix('-')?;

    if remainder.is_empty() {
        return None;
    }

    let tag = if let Some((before, _)) = remainder.rsplit_once("-of-") {
        before.rsplit_once('-').map_or(before, |(t, _)| t)
    } else {
        remainder
    };

    (!tag.is_empty()).then(|| tag.to_string())
}

fn parse_model_dir_name(dir_name: &str) -> Option<(String, String)> {
    let stripped = dir_name.strip_prefix("models--")?;
    let (owner, model) = stripped.split_once("--")?;
    Some((owner.to_string(), model.to_string()))
}

/// Checks which downloaded models have outdated files by comparing commit
/// hashes.
///
/// Groups models by repo so that only one resolve-info HEAD request is
/// made per repo, regardless of how many tags are installed.
pub async fn check_outdated_models(
    hub_dir: Option<PathBuf>,
) -> Result<Vec<OutdatedModelInfo>, PacaError> {
    let hub = HubLayout::open(hub_dir)?;

    let client = Client::builder()
        .default_headers(default_headers())
        .build()?;
    let head_client = resolve_client()?;
    let endpoint = model_endpoint();
    let mut outdated_models = Vec::new();

    let models = list_models(Some(hub.root().to_path_buf()))?;

    let mut repo_outdated: HashMap<String, bool> = HashMap::new();

    for model_ref in &models {
        let repo = model_ref.repo();

        let is_outdated = if let Some(&cached) = repo_outdated.get(&repo) {
            cached
        } else {
            let outdated =
                repo_outdated_check(&client, &head_client, endpoint, &hub, model_ref).await?;
            repo_outdated.insert(repo, outdated);
            outdated
        };

        if !is_outdated {
            continue;
        }

        let manifest = fetch_manifest(&client, model_ref).await?;
        let paths = hub.model(model_ref);
        let local_commit = paths.read_ref();
        let snapshot_path = paths.snapshot(local_commit.as_deref().unwrap_or(""));

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

async fn repo_outdated_check(
    client: &Client,
    head_client: &Client,
    endpoint: &str,
    hub: &HubLayout,
    model_ref: &ModelRef,
) -> Result<bool, PacaError> {
    let manifest = fetch_manifest(client, model_ref).await?;
    let Some(first_file) = manifest.gguf_files.first() else {
        return Ok(false);
    };

    let url = format!(
        "{}/{}/resolve/main/{}",
        endpoint,
        model_ref.repo(),
        first_file.filename
    );

    let local_commit = hub.model(model_ref).read_ref();

    match fetch_resolve_info(head_client, &url).await {
        Ok(info) => Ok(local_commit.as_deref() != Some(&info.commit_hash)),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_ref(s: &str) -> ModelRef {
        s.parse().unwrap()
    }

    #[test]
    fn default_hub_dir_ends_in_huggingface_hub() {
        assert!(
            default_hub_dir()
                .unwrap()
                .ends_with(".cache/huggingface/hub")
        );
    }

    #[test]
    fn hub_layout_creates_override_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("fresh");
        let hub = HubLayout::open(Some(root.clone())).unwrap();
        assert!(root.is_dir());
        assert_eq!(hub.root(), root);
    }

    #[test]
    fn model_dir_name_formats_correctly() {
        assert_eq!(
            model_dir_name(&model_ref("unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL")),
            "models--unsloth--GLM-4.7-Flash-GGUF"
        );
    }

    #[test]
    fn model_paths_derive_from_hub_and_ref() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr = model_ref("unsloth/GLM-4.7-Flash-GGUF:Q2_K_XL");
        let paths = hub.model(&mr);

        let base = dir.path().join("models--unsloth--GLM-4.7-Flash-GGUF");
        assert_eq!(paths.blobs(), base.join("blobs"));
        assert_eq!(paths.refs(), base.join("refs"));
        assert_eq!(paths.snapshots(), base.join("snapshots"));
        assert_eq!(paths.ref_main(), base.join("refs").join("main"));
    }

    #[test]
    fn save_and_read_ref() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr = model_ref("owner/model-GGUF:Q4");

        hub.model(&mr).save_ref("abc123commit").unwrap();
        assert_eq!(hub.model(&mr).read_ref().as_deref(), Some("abc123commit"));
    }

    #[test]
    fn read_ref_returns_none_when_no_ref_file() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        assert_eq!(
            hub.model(&model_ref("owner/model-GGUF:Q4")).read_ref(),
            None
        );
    }

    #[test]
    fn blob_exists_returns_true_when_blob_exists() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        let mr = model_ref("owner/model-GGUF:Q4");
        let paths = hub.model(&mr);
        fs::create_dir_all(paths.blobs()).unwrap();
        fs::write(paths.blob("abcdef1234"), b"data").unwrap();

        assert!(paths.blob_exists("abcdef1234"));
    }

    #[test]
    fn blob_exists_returns_false_when_blob_missing() {
        let dir = tempfile::tempdir().unwrap();
        let hub = HubLayout::open(Some(dir.path().to_path_buf())).unwrap();
        assert!(
            !hub.model(&model_ref("owner/model-GGUF:Q4"))
                .blob_exists("abcdef1234")
        );
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
        assert_eq!(result[0].owner, "owner");
        assert_eq!(result[0].model, "model-GGUF");
        assert_eq!(result[0].tag, "Q4");
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
        assert_eq!(result[0].tag, "BF16");
        assert_eq!(result[1].tag, "Q4_K_M");
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
        assert_eq!(result[0].tag, "BF16");
    }
}
