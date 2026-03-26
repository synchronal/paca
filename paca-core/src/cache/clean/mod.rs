use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cache::{cache_filename, extract_model_ref, get_cache_dir};
use crate::error::PacaError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CleanReason {
    IncompleteDownload,
    IncompleteManifest,
    OrphanedEtag,
    OrphanedGguf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemovedFile {
    pub path: PathBuf,
    pub reason: CleanReason,
}

#[derive(Debug)]
pub struct CleanResult {
    pub removed_files: Vec<RemovedFile>,
}

#[derive(Deserialize)]
struct SavedManifest {
    #[serde(rename = "ggufFile")]
    gguf_file: Option<SavedGgufFileInfo>,
}

#[derive(Deserialize)]
struct SavedGgufFileInfo {
    rfilename: String,
}

fn expand_shard_filenames(filename: &str) -> Vec<String> {
    let Some(stem) = filename.strip_suffix(".gguf") else {
        return vec![filename.to_string()];
    };

    let Some((before_of, total_str)) = stem.rsplit_once("-of-") else {
        return vec![filename.to_string()];
    };

    let Ok(total) = total_str.parse::<usize>() else {
        return vec![filename.to_string()];
    };

    let Some((base, shard_str)) = before_of.rsplit_once('-') else {
        return vec![filename.to_string()];
    };

    let width = shard_str.len();

    (1..=total)
        .map(|i| format!("{base}-{i:0>width$}-of-{total_str}.gguf"))
        .collect()
}

fn expected_cache_filenames(manifest_path: &Path) -> Result<Vec<String>, PacaError> {
    let model_ref = extract_model_ref(manifest_path)?;
    let content = fs::read_to_string(manifest_path).map_err(PacaError::FileWrite)?;
    let saved: SavedManifest = serde_json::from_str(&content)?;

    let rfilename = match saved.gguf_file {
        Some(info) => info.rfilename,
        None => return Ok(Vec::new()),
    };

    let filenames = expand_shard_filenames(&rfilename);
    Ok(filenames
        .iter()
        .map(|f| cache_filename(&model_ref, f))
        .collect())
}

pub fn clean_cache(cache_dir: Option<PathBuf>) -> Result<CleanResult, PacaError> {
    let cache_dir = match cache_dir {
        Some(dir) => dir,
        None => get_cache_dir()?,
    };

    if !cache_dir.exists() {
        return Ok(CleanResult {
            removed_files: Vec::new(),
        });
    }

    let mut manifests: Vec<PathBuf> = Vec::new();
    let mut ggufs: HashSet<String> = HashSet::new();
    let mut etags: HashSet<String> = HashSet::new();

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();

        if name.starts_with("manifest=") && name.ends_with(".json") {
            manifests.push(entry.path());
        } else if name.ends_with(".gguf") {
            ggufs.insert(name);
        } else if name.ends_with(".etag") {
            etags.insert(name);
        }
    }

    let mut removed_files = Vec::new();
    let mut claimed_ggufs: HashSet<String> = HashSet::new();
    let mut removed_ggufs: HashSet<String> = HashSet::new();

    for manifest_path in &manifests {
        let expected = match expected_cache_filenames(manifest_path) {
            Ok(filenames) => filenames,
            Err(_) => {
                // Unparseable manifest — remove it
                fs::remove_file(manifest_path).map_err(PacaError::FileDelete)?;
                removed_files.push(RemovedFile {
                    path: manifest_path.clone(),
                    reason: CleanReason::IncompleteManifest,
                });
                continue;
            }
        };

        let all_present = expected.iter().all(|f| ggufs.contains(f));

        if all_present {
            for f in &expected {
                claimed_ggufs.insert(f.clone());
            }
        } else {
            // Incomplete manifest — remove manifest + partial GGUFs + etags
            fs::remove_file(manifest_path).map_err(PacaError::FileDelete)?;
            removed_files.push(RemovedFile {
                path: manifest_path.clone(),
                reason: CleanReason::IncompleteManifest,
            });

            for f in &expected {
                if ggufs.contains(f) {
                    let gguf_path = cache_dir.join(f);
                    fs::remove_file(&gguf_path).map_err(PacaError::FileDelete)?;
                    removed_files.push(RemovedFile {
                        path: gguf_path,
                        reason: CleanReason::IncompleteDownload,
                    });
                    removed_ggufs.insert(f.clone());
                }

                let etag_name = format!("{}.etag", f);
                if etags.contains(&etag_name) {
                    let etag_path = cache_dir.join(&etag_name);
                    fs::remove_file(&etag_path).map_err(PacaError::FileDelete)?;
                    removed_files.push(RemovedFile {
                        path: etag_path,
                        reason: CleanReason::IncompleteDownload,
                    });
                }
            }
        }
    }

    // Orphaned GGUFs — not claimed by any complete manifest
    for gguf_name in &ggufs {
        if claimed_ggufs.contains(gguf_name) || removed_ggufs.contains(gguf_name) {
            continue;
        }

        let gguf_path = cache_dir.join(gguf_name);
        fs::remove_file(&gguf_path).map_err(PacaError::FileDelete)?;
        removed_files.push(RemovedFile {
            path: gguf_path,
            reason: CleanReason::OrphanedGguf,
        });
        removed_ggufs.insert(gguf_name.clone());

        let etag_name = format!("{}.etag", gguf_name);
        if etags.contains(&etag_name) {
            let etag_path = cache_dir.join(&etag_name);
            fs::remove_file(&etag_path).map_err(PacaError::FileDelete)?;
            removed_files.push(RemovedFile {
                path: etag_path,
                reason: CleanReason::OrphanedEtag,
            });
        }
    }

    // Dangling etags — no corresponding GGUF
    for etag_name in &etags {
        let gguf_name = etag_name.strip_suffix(".etag").unwrap_or(etag_name);
        if ggufs.contains(gguf_name) && !removed_ggufs.contains(gguf_name) {
            continue;
        }

        let etag_path = cache_dir.join(etag_name);
        if etag_path.exists() {
            fs::remove_file(&etag_path).map_err(PacaError::FileDelete)?;
            removed_files.push(RemovedFile {
                path: etag_path,
                reason: CleanReason::OrphanedEtag,
            });
        }
    }

    removed_files.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(CleanResult { removed_files })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use crate::cache::save_etag;

    // --- expand_shard_filenames tests ---

    #[test]
    fn expand_shard_filenames_returns_single_for_non_sharded() {
        let result = expand_shard_filenames("model-Q4_K_M.gguf");
        assert_eq!(result, vec!["model-Q4_K_M.gguf"]);
    }

    #[test]
    fn expand_shard_filenames_returns_all_for_two_shards() {
        let result = expand_shard_filenames("BF16/Model-BF16-00001-of-00002.gguf");
        assert_eq!(
            result,
            vec![
                "BF16/Model-BF16-00001-of-00002.gguf",
                "BF16/Model-BF16-00002-of-00002.gguf",
            ]
        );
    }

    #[test]
    fn expand_shard_filenames_returns_all_for_fifteen_shards() {
        let result = expand_shard_filenames("BF16/GLM-4.7-BF16-00001-of-00015.gguf");
        assert_eq!(result.len(), 15);
        assert_eq!(result[0], "BF16/GLM-4.7-BF16-00001-of-00015.gguf");
        assert_eq!(result[7], "BF16/GLM-4.7-BF16-00008-of-00015.gguf");
        assert_eq!(result[14], "BF16/GLM-4.7-BF16-00015-of-00015.gguf");
    }

    // --- clean_cache integration tests ---

    fn write_manifest(dir: &Path, owner: &str, model: &str, tag: &str, rfilename: &str) {
        let manifest_name = format!("manifest={owner}={model}={tag}.json");
        let json = format!(r#"{{"ggufFile":{{"rfilename":"{rfilename}","size":1024}}}}"#);
        fs::write(dir.join(manifest_name), json).unwrap();
    }

    fn write_gguf(dir: &Path, filename: &str) {
        fs::write(dir.join(filename), b"fake gguf data").unwrap();
    }

    fn file_exists(dir: &Path, filename: &str) -> bool {
        dir.join(filename).exists()
    }

    #[test]
    fn clean_cache_empty_dir_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_nonexistent_dir_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does-not-exist");
        let result = clean_cache(Some(nonexistent)).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_complete_single_file_model_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();

        write_manifest(dir.path(), "owner", "model-GGUF", "Q4", "model-Q4.gguf");
        write_gguf(dir.path(), "owner_model-GGUF_model-Q4.gguf");
        save_etag(dir.path(), "owner_model-GGUF_model-Q4.gguf", "\"etag1\"").unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
        assert!(file_exists(dir.path(), "manifest=owner=model-GGUF=Q4.json"));
        assert!(file_exists(dir.path(), "owner_model-GGUF_model-Q4.gguf"));
        assert!(file_exists(
            dir.path(),
            "owner_model-GGUF_model-Q4.gguf.etag"
        ));
    }

    #[test]
    fn clean_cache_complete_sharded_model_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();

        write_manifest(
            dir.path(),
            "owner",
            "model-GGUF",
            "BF16",
            "BF16/model-BF16-00001-of-00002.gguf",
        );
        write_gguf(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00001-of-00002.gguf",
        );
        write_gguf(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00002-of-00002.gguf",
        );

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_incomplete_manifest_removes_manifest_and_partial_files() {
        let dir = tempfile::tempdir().unwrap();

        // Manifest expects 2 shards but only 1 exists
        write_manifest(
            dir.path(),
            "owner",
            "model-GGUF",
            "BF16",
            "BF16/model-BF16-00001-of-00002.gguf",
        );
        write_gguf(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00001-of-00002.gguf",
        );
        save_etag(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00001-of-00002.gguf",
            "\"etag1\"",
        )
        .unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        assert!(!file_exists(
            dir.path(),
            "manifest=owner=model-GGUF=BF16.json"
        ));
        assert!(!file_exists(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00001-of-00002.gguf"
        ));
        assert!(!file_exists(
            dir.path(),
            "owner_model-GGUF_BF16_model-BF16-00001-of-00002.gguf.etag"
        ));

        assert_eq!(result.removed_files.len(), 3);

        let reasons: Vec<&CleanReason> = result.removed_files.iter().map(|r| &r.reason).collect();
        assert!(reasons.contains(&&CleanReason::IncompleteManifest));
        assert!(reasons.contains(&&CleanReason::IncompleteDownload));
    }

    #[test]
    fn clean_cache_orphaned_gguf_removes_gguf_and_etag() {
        let dir = tempfile::tempdir().unwrap();

        // GGUF with no manifest
        write_gguf(dir.path(), "owner_model-GGUF_model-Q4.gguf");
        save_etag(dir.path(), "owner_model-GGUF_model-Q4.gguf", "\"etag1\"").unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        assert!(!file_exists(dir.path(), "owner_model-GGUF_model-Q4.gguf"));
        assert!(!file_exists(
            dir.path(),
            "owner_model-GGUF_model-Q4.gguf.etag"
        ));

        assert_eq!(result.removed_files.len(), 2);

        let reasons: Vec<&CleanReason> = result.removed_files.iter().map(|r| &r.reason).collect();
        assert!(reasons.contains(&&CleanReason::OrphanedGguf));
        assert!(reasons.contains(&&CleanReason::OrphanedEtag));
    }

    #[test]
    fn clean_cache_mixed_scenario_removes_only_stale_files() {
        let dir = tempfile::tempdir().unwrap();

        // Complete model — should be kept
        write_manifest(dir.path(), "owner", "good-GGUF", "Q4", "model-Q4.gguf");
        write_gguf(dir.path(), "owner_good-GGUF_model-Q4.gguf");
        save_etag(dir.path(), "owner_good-GGUF_model-Q4.gguf", "\"good\"").unwrap();

        // Incomplete manifest — should be removed
        write_manifest(
            dir.path(),
            "owner",
            "bad-GGUF",
            "BF16",
            "BF16/model-BF16-00001-of-00002.gguf",
        );
        write_gguf(
            dir.path(),
            "owner_bad-GGUF_BF16_model-BF16-00001-of-00002.gguf",
        );

        // Orphaned GGUF — should be removed
        write_gguf(dir.path(), "owner_orphan-GGUF_model.gguf");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        // Complete model untouched
        assert!(file_exists(dir.path(), "manifest=owner=good-GGUF=Q4.json"));
        assert!(file_exists(dir.path(), "owner_good-GGUF_model-Q4.gguf"));
        assert!(file_exists(
            dir.path(),
            "owner_good-GGUF_model-Q4.gguf.etag"
        ));

        // Incomplete manifest removed
        assert!(!file_exists(
            dir.path(),
            "manifest=owner=bad-GGUF=BF16.json"
        ));
        assert!(!file_exists(
            dir.path(),
            "owner_bad-GGUF_BF16_model-BF16-00001-of-00002.gguf"
        ));

        // Orphaned GGUF removed
        assert!(!file_exists(dir.path(), "owner_orphan-GGUF_model.gguf"));

        // Should have removed: manifest + 1 partial GGUF + 1 orphaned GGUF = 3
        assert_eq!(result.removed_files.len(), 3);
    }

    #[test]
    fn clean_cache_unparseable_manifest_removes_it() {
        let dir = tempfile::tempdir().unwrap();

        let manifest_name = "manifest=owner=broken=Q4.json";
        fs::write(dir.path().join(manifest_name), "not valid json!!!").unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        assert!(!file_exists(dir.path(), manifest_name));
        assert_eq!(result.removed_files.len(), 1);
        assert_eq!(
            result.removed_files[0].reason,
            CleanReason::IncompleteManifest
        );
    }

    #[test]
    fn clean_cache_dangling_etag_without_gguf_removed() {
        let dir = tempfile::tempdir().unwrap();

        // Etag with no corresponding GGUF
        save_etag(dir.path(), "owner_model-GGUF_model-Q4.gguf", "\"etag1\"").unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        assert!(!file_exists(
            dir.path(),
            "owner_model-GGUF_model-Q4.gguf.etag"
        ));
        assert_eq!(result.removed_files.len(), 1);
        assert_eq!(result.removed_files[0].reason, CleanReason::OrphanedEtag);
    }
}
