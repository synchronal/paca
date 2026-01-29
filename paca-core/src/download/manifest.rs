use reqwest::blocking::Client;
use serde::Deserialize;

use crate::error::DownloadError;

use super::USER_AGENT;
use super::endpoint::get_model_endpoint;
use super::model_ref::ModelRef;

#[derive(Debug, Deserialize)]
struct ManifestResponse {
    #[serde(rename = "ggufFile")]
    gguf_file: Option<GgufFileInfo>,
}

#[derive(Debug, Deserialize)]
struct GgufFileInfo {
    rfilename: String,
    size: u64,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// List of GGUF files to download
    pub gguf_files: Vec<GgufFile>,
    /// Raw JSON response from HuggingFace
    pub raw_json: String,
}

/// Fetches the model manifest from HuggingFace, handling both single and sharded files
pub fn fetch_manifest(client: &Client, model_ref: &ModelRef) -> Result<Manifest, DownloadError> {
    let endpoint = get_model_endpoint();
    let url = format!(
        "{}/v2/{}/manifests/{}",
        endpoint,
        model_ref.repo(),
        model_ref.tag
    );

    let response = client
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .send()?
        .error_for_status()?;

    let raw_json = response.text()?;
    let manifest_response: ManifestResponse = serde_json::from_str(&raw_json)?;

    let gguf_file_info = manifest_response
        .gguf_file
        .ok_or(DownloadError::NoGgufFile)?;

    let gguf_files = match shard_count(&gguf_file_info.rfilename) {
        Some(_) => fetch_tree_files(client, &endpoint, model_ref, &gguf_file_info.rfilename)?,
        None => vec![GgufFile {
            filename: gguf_file_info.rfilename,
            size: gguf_file_info.size,
        }],
    };

    Ok(Manifest {
        gguf_files,
        raw_json,
    })
}

/// Fetches sharded GGUF files from the HuggingFace tree API
fn fetch_tree_files(
    client: &Client,
    endpoint: &str,
    model_ref: &ModelRef,
    rfilename: &str,
) -> Result<Vec<GgufFile>, DownloadError> {
    let subdir = rfilename.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");

    let url = format!(
        "{}/api/models/{}/tree/main/{}",
        endpoint,
        model_ref.repo(),
        subdir
    );

    let response = client
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .send()?
        .error_for_status()?;

    let entries: Vec<TreeEntry> = response.json()?;

    let mut gguf_files: Vec<GgufFile> = entries
        .into_iter()
        .filter(|entry| entry.path.ends_with(".gguf"))
        .map(|entry| GgufFile {
            filename: entry.path,
            size: entry.size,
        })
        .collect();

    gguf_files.sort_by_key(|file| file.filename.clone());

    Ok(gguf_files)
}

fn shard_count(filename: &str) -> Option<usize> {
    let stem = filename.strip_suffix(".gguf")?;
    let of_part = stem.rsplit_once("-of-")?;
    of_part.1.parse::<usize>().ok()
}

/// Generates a unique filename for the model manifest
pub fn manifest_filename(model_ref: &ModelRef) -> String {
    format!(
        "manifest={}={}={}.json",
        model_ref.owner, model_ref.model, model_ref.tag
    )
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
            raw_json: "{}".to_string(),
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
            raw_json: "{}".to_string(),
        };
        assert_eq!(manifest.gguf_files.len(), 2);
        assert_eq!(manifest.gguf_files[0].filename, "file-00001-of-00002.gguf");
        assert_eq!(manifest.gguf_files[1].filename, "file-00002-of-00002.gguf");
    }

    #[test]
    fn manifest_filename_formats_correctly() {
        let model_ref: ModelRef = "unsloth/GLM-4.7-Flash-GGUF:BF16".parse().unwrap();
        let filename = manifest_filename(&model_ref);
        assert_eq!(filename, "manifest=unsloth=GLM-4.7-Flash-GGUF=BF16.json");
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
}
