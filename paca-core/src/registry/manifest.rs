use std::fmt;

use reqwest::Client;
use serde::Deserialize;

use crate::error::PacaError;
use crate::model::ModelRef;
use crate::registry::endpoint::get_model_endpoint;

#[derive(Debug, Deserialize)]
struct TreeEntry {
    path: String,
    size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufFile {
    /// The filename of the GGUF file
    pub filename: String,
    /// The size of the file in bytes
    pub size: u64,
}

impl fmt::Display for GgufFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.filename)
    }
}

impl From<TreeEntry> for GgufFile {
    fn from(entry: TreeEntry) -> Self {
        Self {
            filename: entry.path,
            size: entry.size,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// List of GGUF files to download
    pub gguf_files: Vec<GgufFile>,
}

/// Fetches the model manifest from HuggingFace, handling both single and sharded files
pub async fn fetch_manifest(client: &Client, model_ref: &ModelRef) -> Result<Manifest, PacaError> {
    let endpoint = get_model_endpoint();
    let url = format!(
        "{}/v2/{}/manifests/{}",
        endpoint,
        model_ref.repo(),
        model_ref.tag
    );

    let response = client.get(&url).send().await?.error_for_status()?;

    let parsed: serde_json::Value = response.json().await?;
    let discovered = collect_manifest_files(&parsed);

    if discovered.is_empty() {
        return Err(PacaError::NoFiles);
    }

    let mut gguf_files = Vec::new();
    for file in discovered {
        match shard_count(&file.filename) {
            Some(_) => {
                gguf_files
                    .extend(fetch_tree_files(client, &endpoint, model_ref, &file.filename).await?);
            }
            None => gguf_files.push(file),
        }
    }

    Ok(Manifest { gguf_files })
}

/// Walks top-level JSON entries and collects any object with `rfilename` (String) + `size` (u64)
fn collect_manifest_files(value: &serde_json::Value) -> Vec<GgufFile> {
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };

    let mut files: Vec<GgufFile> = Vec::new();

    for (_key, entry) in obj {
        let Some(entry_obj) = entry.as_object() else {
            continue;
        };

        let rfilename = entry_obj
            .get("rfilename")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let size = entry_obj.get("size").and_then(|v| v.as_u64());

        if let (Some(filename), Some(size)) = (rfilename, size) {
            files.push(GgufFile { filename, size });
        }
    }

    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    files
}

/// Fetches sharded GGUF files from the HuggingFace tree API
async fn fetch_tree_files(
    client: &Client,
    endpoint: &str,
    model_ref: &ModelRef,
    rfilename: &str,
) -> Result<Vec<GgufFile>, PacaError> {
    let subdir = rfilename.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");

    let url = format!(
        "{}/api/models/{}/tree/main/{}",
        endpoint,
        model_ref.repo(),
        subdir
    );

    let response = client.get(&url).send().await?.error_for_status()?;

    let entries: Vec<TreeEntry> = response.json().await?;

    let mut gguf_files: Vec<GgufFile> = entries
        .into_iter()
        .filter(|entry| entry.path.ends_with(".gguf"))
        .map(GgufFile::from)
        .collect();

    gguf_files.sort_by(|a, b| a.filename.cmp(&b.filename));

    Ok(gguf_files)
}

fn shard_count(filename: &str) -> Option<usize> {
    let stem = filename.strip_suffix(".gguf")?;
    let of_part = stem.rsplit_once("-of-")?;
    of_part.1.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_struct_holds_single_file() {
        let manifest = Manifest {
            gguf_files: vec![GgufFile {
                filename: "model.gguf".to_string(),
                size: 1024,
            }],
        };
        assert_eq!(manifest.gguf_files.len(), 1);
        assert_eq!(manifest.gguf_files[0].filename, "model.gguf");
        assert_eq!(manifest.gguf_files[0].size, 1024);
    }

    #[test]
    fn manifest_struct_holds_multiple_files() {
        let manifest = Manifest {
            gguf_files: vec![
                GgufFile {
                    filename: "file-00001-of-00002.gguf".to_string(),
                    size: 1024,
                },
                GgufFile {
                    filename: "file-00002-of-00002.gguf".to_string(),
                    size: 2048,
                },
            ],
        };
        assert_eq!(manifest.gguf_files.len(), 2);
        assert_eq!(manifest.gguf_files[0].filename, "file-00001-of-00002.gguf");
        assert_eq!(manifest.gguf_files[1].filename, "file-00002-of-00002.gguf");
    }

    #[test]
    fn shard_count_returns_none_for_single_file() {
        assert_eq!(shard_count("model.gguf"), None);
    }

    #[test]
    fn shard_count_returns_none_for_single_file_with_directory() {
        assert_eq!(shard_count("Q4_K_M/model-Q4_K_M.gguf"), None);
    }

    #[test]
    fn shard_count_returns_total_for_sharded_file() {
        assert_eq!(
            shard_count("BF16/GLM-4.7-BF16-00001-of-00015.gguf"),
            Some(15)
        );
    }

    #[test]
    fn shard_count_returns_total_for_two_shards() {
        assert_eq!(shard_count("BF16/Model-BF16-00001-of-00002.gguf"), Some(2));
    }

    #[test]
    fn gguf_file_displays_filename() {
        let file = GgufFile {
            filename: "model-Q4_K_M.gguf".to_string(),
            size: 4096,
        };
        assert_eq!(file.to_string(), "model-Q4_K_M.gguf");
    }

    #[test]
    fn collect_manifest_files_finds_gguf_file() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"ggufFile":{"rfilename":"model-Q4.gguf","size":1024}}"#)
                .unwrap();
        let files = collect_manifest_files(&json);
        assert_eq!(
            files,
            vec![GgufFile {
                filename: "model-Q4.gguf".to_string(),
                size: 1024
            }]
        );
    }

    #[test]
    fn collect_manifest_files_finds_multiple_file_types() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "ggufFile":{"rfilename":"model.gguf","size":1024},
                "mmprojFile":{"rfilename":"mmproj-BF16.gguf","size":512}
            }"#,
        )
        .unwrap();
        let files = collect_manifest_files(&json);
        assert_eq!(
            files,
            vec![
                GgufFile {
                    filename: "mmproj-BF16.gguf".to_string(),
                    size: 512
                },
                GgufFile {
                    filename: "model.gguf".to_string(),
                    size: 1024
                },
            ]
        );
    }

    #[test]
    fn collect_manifest_files_skips_entries_without_rfilename() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "ggufFile":{"rfilename":"model.gguf","size":1024},
                "config":{"key":"value"}
            }"#,
        )
        .unwrap();
        let files = collect_manifest_files(&json);
        assert_eq!(
            files,
            vec![GgufFile {
                filename: "model.gguf".to_string(),
                size: 1024
            }]
        );
    }

    #[test]
    fn collect_manifest_files_returns_empty_for_no_files() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"config":{"key":"value"}}"#).unwrap();
        let files = collect_manifest_files(&json);
        assert!(files.is_empty());
    }

    #[test]
    fn gguf_file_from_tree_entry() {
        let entry = TreeEntry {
            path: "BF16/model-00001-of-00002.gguf".to_string(),
            size: 2048,
        };
        let file = GgufFile::from(entry);
        assert_eq!(file.filename, "BF16/model-00001-of-00002.gguf");
        assert_eq!(file.size, 2048);
    }
}
